import re
import os
import time
import json
import requests
from typing import List, Dict, Optional
from PyPDF2 import PdfReader

# ============================================================
# 0. 전역 설정
# ============================================================

# User-Agent 설정 (봇 차단 방지)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "mailto": "your_email@example.com"  # Crossref/OpenAlex 등에서 권장
}

DOI_REGEX = r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b'
YEAR_REGEX = r"(19|20)\d{2}"

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"

DOWNLOAD_DIR = r"I:\내 드라이브\hogan_foot_force"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ============================================================
# 1. PDF 텍스트 추출 및 Reference 섹션 감지
# ============================================================

def extract_arxiv_id(text: str) -> Optional[str]:
    """ 
    텍스트에서 arXiv ID 추출 
    (지원 형식: arXiv:2402.05952, abs/2402.05952, pdf/2402.05952 등)
    """
    match = re.search(r'(?:arXiv:|abs/|pdf/)?(\d{4}\.\d{4,5})(?:v\d+)?', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def extract_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages

def detect_reference_start(text: str) -> int:
    # "References", "Bibliography" 등을 대소문자 무시하고 찾기
    m = re.search(r"^\s*(References|Bibliography)\s*$", text, re.IGNORECASE | re.MULTILINE)
    return m.end() if m else -1

def extract_reference_text_from_pdf(pdf_path: str) -> str:
    """
    (구) References 섹션 이후 텍스트 덩어리만 추출.
    지금은 아래의 extract_references_structured에서
    PDF 전체를 다시 읽어서 쓰므로, 다른 PDF용으로 남겨둠.
    """
    pages = extract_pdf_pages(pdf_path)
    full_text = "\n".join(pages)
    
    cutoff = int(len(full_text) * 0.5)
    search_area = full_text[cutoff:]
    
    match = re.search(r"(?:\n|^)\s*(References|Bibliography|REFERENCES)\s*(?:\n|$)", search_area)
    
    if match:
        start_idx = cutoff + match.end()
        return full_text[start_idx:].strip()
    
    # 못 찾은 경우 마지막 2페이지만 반환 (Fallback)
    return "\n".join(pages[-2:])

# ============================================================
# 1-1. (새로 추가) [번호] 기준으로 레퍼런스 + 제목 + arXiv 블록 추출
# ============================================================

def extract_references_structured(pdf_path: str) -> List[Dict]:
    """
    이 함수가 네가 원한 기능:
    - PDF 전체 텍스트를 읽고
    - 'References' 이후 부분만 잘라서
    - [1] ... [2] ... 형태로 레퍼런스 분리
    - 각 레퍼런스에서
        * 연도 뒤에 나오는 논문 제목 추출 (title_from_ref)
        * 'arXiv:2401.15569 [cs.**]' 전체 블록 추출 (arxiv_block)
        * 기존 파이프라인이 쓰는 raw, doi, year, has_doi도 같이 반환
    """
    reader = PdfReader(pdf_path)

    # 1. PDF 전체 텍스트 추출
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # 2. 줄바꿈에 의한 하이픈 분리 단어 복구 & 공백 정리
    full_text = re.sub(r"-\n", "", full_text)   # "High-\nFidelity" -> "HighFidelity"
    full_text = full_text.replace("\n", " ")

    # 3. References 섹션 이후만 사용
    try:
        start = full_text.index("References")
        refs_text = full_text[start:]
        refs_text = refs_text.split("References", 1)[1]
    except ValueError:
        # 못 찾으면 기존 방식 시도
        idx = detect_reference_start(full_text)
        if idx != -1:
            refs_text = full_text[idx:]
        else:
            refs_text = full_text  # 최후의 fallback

    # 4. [1] [2] [3] ... 패턴으로 레퍼런스를 분리
    parts = re.split(r"\[(\d+)\]", refs_text)

    parsed_refs: List[Dict] = []

    # 4. [1] [2] ... 패턴으로 먼저 시도
    parts = re.split(r"\[(\d+)\]", refs_text)

    if len(parts) > 1:
        # 기존 [n] 방식
        for i in range(1, len(parts), 2):
            idx_str = parts[i]
            content = parts[i + 1].strip()
            if not content:
                continue

            index = int(idx_str)

            doi_match = re.search(DOI_REGEX, content, re.IGNORECASE)
            year_match = re.search(YEAR_REGEX, content)

            # 제목 / arxiv 추출 로직 그대로 사용
            title_from_ref = None
            year_dot_match = re.search(r"\b(19|20)\d{2}\.\s", content)
            if year_dot_match:
                after_year = content[year_dot_match.end():]
                after_year = re.sub(r"\s+", " ", after_year)
                if "." in after_year:
                    title_from_ref = after_year.split(".", 1)[0].strip()
                else:
                    title_from_ref = after_year.strip()
                title_from_ref = title_from_ref.rstrip(".")

            arxiv_block = None
            arxiv_match = re.search(
                r"arXiv:\s*\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\]",
                content
            )
            if arxiv_match:
                arxiv_block = arxiv_match.group(0)

            parsed_refs.append({
                "index": index,
                "raw": content,
                "doi": doi_match.group(0) if doi_match else None,
                "year": year_match.group(0) if year_match else None,
                "has_doi": bool(doi_match),
                "title_from_ref": title_from_ref,
                "arxiv_block": arxiv_block,
            })

    else:
        # [n] 형식이 아니면, 예전 줄 기반 splitter 사용 (1. 형식 포함)
        items = split_reference_items(refs_text)
        for index, content in enumerate(items, start=1):
            content = content.strip()
            if not content:
                continue

            doi_match = re.search(DOI_REGEX, content, re.IGNORECASE)
            year_match = re.search(YEAR_REGEX, content)

            title_from_ref = None
            year_dot_match = re.search(r"\b(19|20)\d{2}\.\s", content)
            if year_dot_match:
                after_year = content[year_dot_match.end():]
                after_year = re.sub(r"\s+", " ", after_year)
                if "." in after_year:
                    title_from_ref = after_year.split(".", 1)[0].strip()
                else:
                    title_from_ref = after_year.strip()
                title_from_ref = title_from_ref.rstrip(".")

            arxiv_block = None
            arxiv_match = re.search(
                r"arXiv:\s*\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\]",
                content
            )
            if arxiv_match:
                arxiv_block = arxiv_match.group(0)

            parsed_refs.append({
                "index": index,
                "raw": content,
                "doi": doi_match.group(0) if doi_match else None,
                "year": year_match.group(0) if year_match else None,
                "has_doi": bool(doi_match),
                "title_from_ref": title_from_ref,
                "arxiv_block": arxiv_block,
            })

    return parsed_refs


# ============================================================
# 2. Reference 항목 분리 (기존 버전 - 지금은 안 씀)
# ============================================================

def split_reference_items(ref_text: str) -> List[str]:
    """
    (구) 레퍼런스 텍스트 덩어리를 개별 레퍼런스로 분리.
    지금은 extract_references_structured를 쓰므로,
    다른 PDF용으로 남겨두기만 함.
    """
    lines = ref_text.split('\n')
    items = []
    current_item = []

    new_item_pattern = re.compile(r'^(\[\d+\]|\d+\.|[A-Z][a-zA-Z\-\']+(?:,\s+[A-Z]\.|[\s,]+et\s+al))')

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if re.match(r'^\d+:\d+$', line) or re.match(r'^Peng et al\.$', line):
            continue

        if new_item_pattern.match(line):
            if current_item:
                items.append(" ".join(current_item))
            current_item = [line]
        else:
            current_item.append(line)

    if current_item:
        items.append(" ".join(current_item))
    
    return items

def parse_reference_item(item: str) -> Dict:
    """
    (구) 문자열 하나에서 raw, doi, year만 뽑던 함수.
    지금은 extract_references_structured에서 바로 dict를 만들기 때문에,
    다른 용도로 쓸 수 있게 남겨둠.
    """
    doi = re.search(DOI_REGEX, item, re.IGNORECASE)
    year = re.search(YEAR_REGEX, item)
    return {
        "raw": item,
        "doi": doi.group(0) if doi else None,
        "year": year.group(0) if year else None,
        "has_doi": bool(doi),
    }


# ============================================================
# 3. 검색 쿼리 정제 (핵심 기능)
# ============================================================

def clean_query_aggressive(raw_ref: str) -> str:
    text = raw_ref.replace('\n', ' ').strip()
    text = re.sub(r'^\[\d+\]\s*', '', text)  # [1] 제거
    text = re.sub(r'^\d+\.\s*', '', text)    # 1. 제거
    
    text = re.sub(r'https?://\S+', '', text)
    
    split_patterns = [
        r'\sIn\s+Proceedings',
        r'\sIn\s+[A-Z]',
        r'\sarXiv:',
        r'\svol\.\s',
        r'\sdoi:',
        r'\sISBN',
    ]
    
    for pattern in split_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break

    parts = text.split('. ')
    
    potential_titles = []
    for part in parts:
        part = part.strip()
        if len(part) < 5: 
            continue
        
        is_author = False
        if "et al" in part or "and " in part:
            if re.search(r'[A-Z]\.', part): 
                is_author = True
        
        if re.match(r'^[\(\[]?\d{4}[\)\]]?$', part):
            continue
            
        if not is_author:
            potential_titles.append(part)
    
    if potential_titles:
        return potential_titles[0]
        
    return text.strip()[:200]


# ============================================================
# 4. API 검색 및 PDF URL 확보 (Semantic Scholar + OpenAlex)
# ============================================================
# ============================================================
# A. arXiv / CrossRef / Unpaywall 헬퍼 함수 추가
# ============================================================

def search_arxiv_by_title(title: str) -> Optional[Dict]:
    """
    arXiv API로 제목 기반 검색.
    - 성공 시: arxiv_id, pdf_url, title, abstract, year 정도를 반환
    - 실패 시: None
    """
    import xml.etree.ElementTree as ET

    if not title:
        return None

    base_url = "http://export.arxiv.org/api/query"
    query = f'ti:"{title}"'
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 1,
    }

    try:
        r = requests.get(base_url, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        # Atom XML 파싱
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        # id에서 arXiv ID 추출 (예: http://arxiv.org/abs/2005.14165v2)
        id_tag = entry.find("atom:id", ns)
        if id_tag is None or "arxiv.org/abs/" not in id_tag.text:
            return None

        arxiv_id = id_tag.text.split("arxiv.org/abs/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # 제목/초록/날짜 등 옵션
        title_tag = entry.find("atom:title", ns)
        summary_tag = entry.find("atom:summary", ns)
        published_tag = entry.find("atom:published", ns)

        out = {
            "source": "arxiv",
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "title": title_tag.text.strip() if title_tag is not None else None,
            "abstract": summary_tag.text.strip() if summary_tag is not None else None,
            "year": None,
        }

        if published_tag is not None and len(published_tag.text) >= 4:
            out["year"] = published_tag.text[:4]

        return out

    except Exception as e:
        print(f"  [Error] arXiv API: {e}")
        return None


def search_crossref_by_title(title: str) -> Optional[Dict]:
    """
    CrossRef로 제목 검색 → DOI 얻기.
    - 성공 시: {"doi": "...", "title": "...", "year": 2020, ...}
    - 실패 시: None
    """
    if not title:
        return None

    try:
        params = {
            "query.title": title,
            "rows": 1,
        }
        r = requests.get(CROSSREF_API, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        items = data.get("message", {}).get("items", [])
        if not items:
            return None

        item = items[0]
        doi = item.get("DOI")
        titles = item.get("title") or []
        item_title = titles[0] if titles else None

        year = None
        if "issued" in item and "date-parts" in item["issued"]:
            try:
                year = item["issued"]["date-parts"][0][0]
            except Exception:
                pass

        return {
            "source": "crossref",
            "doi": doi,
            "title": item_title,
            "year": year,
        }

    except Exception as e:
        print(f"  [Error] CrossRef: {e}")
        return None


def get_unpaywall_pdf(doi: str) -> Optional[Dict]:
    """
    Unpaywall로 DOI 기반 OA PDF URL 검색.
    - 성공 시: {"pdf_url": "...", "url": "..."} 형태 반환
    - 실패 시: None
    """
    if not doi:
        return None

    email = HEADERS.get("mailto", "your_email@example.com")
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": email}

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()

        # best_oa_location이 제일 좋음
        loc = data.get("best_oa_location")
        if not loc and data.get("oa_locations"):
            loc = data["oa_locations"][0]

        if not loc:
            return None

        pdf_url = loc.get("url_for_pdf") or loc.get("url")
        if not pdf_url:
            return None

        return {
            "pdf_url": pdf_url,
            "source": "unpaywall",
        }

    except Exception as e:
        print(f"  [Error] Unpaywall: {e}")
        return None

def search_semantic_scholar(query: str) -> Optional[Dict]:
    try:
        params = {
            "query": query,
            "limit": 1,
            "fields": "title,year,abstract,openAccessPdf,externalIds,url"
        }
        r = requests.get(SEMANTIC_SCHOLAR_API, params=params, headers=HEADERS, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            if data.get("data"):
                paper = data["data"][0]
                pdf_url = None
                if paper.get("openAccessPdf"):
                    pdf_url = paper["openAccessPdf"].get("url")
                
                return {
                    "source": "semantic_scholar",
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "pdf_url": pdf_url,
                    "doi": paper.get("externalIds", {}).get("DOI")
                }
    except Exception as e:
        print(f"  [Error] Semantic Scholar: {e}")
    return None

def search_openalex(query: str) -> Optional[Dict]:
    try:
        params = {"search": query, "per-page": 1}
        r = requests.get(OPENALEX_API, params=params, headers=HEADERS, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                paper = results[0]
                pdf_url = None
                if paper.get("best_oa_location"):
                    pdf_url = paper["best_oa_location"].get("pdf_url")
                
                return {
                    "source": "openalex",
                    "title": paper.get("display_name"),
                    "abstract": None,
                    "pdf_url": pdf_url,
                    "doi": paper.get("doi")
                }
    except Exception as e:
        print(f"  [Error] OpenAlex: {e}")
    return None
SCITEPRESS_DOI_PATTERN = re.compile(r"^10\.5220/00(\d{6})\d+$")

def extract_scitepress_code_from_doi(doi: str) -> Optional[str]:
    """
    SciTePress DOI에서 6자리 paper code 추출.
    예: 10.5220/0012812800003764 -> '128128'
    """
    if not doi:
        return None
    m = SCITEPRESS_DOI_PATTERN.match(doi.strip())
    if not m:
        return None
    return m.group(1)


def build_scitepress_pdf_url(doi: str, year: str) -> Optional[str]:
    """
    SciTePress DOI + 연도 -> PDF URL 구성
    예: (10.5220/0012812800003764, '2024')
      -> https://www.scitepress.org/Papers/2024/128128/128128.pdf
    """
    code = extract_scitepress_code_from_doi(doi)
    if not code or not year:
        return None
    return f"https://www.scitepress.org/Papers/{year}/{code}/{code}.pdf"


def try_scitepress_pdf(doi: str, year: str) -> Optional[str]:
    """
    실제로 URL을 찍어보고 PDF인지 확인까지 해주는 함수.
    """
    url = build_scitepress_pdf_url(doi, year)
    if not url:
        return None

    try:
        # HEAD로 가볍게 체크해도 되고, GET으로 바로 받아도 됨
        r = requests.get(url, headers=HEADERS, stream=True, timeout=10)
        if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", "").lower():
            return url
    except Exception as e:
        print(f"  [SciTePress check error] {e}")

    return None

def get_metadata_and_link_enhanced(ref: Dict) -> Dict:
    """
    검색 전략 우선순위:
    1) 레퍼런스에 직접 있는 arXiv ID 사용
    2) arXiv API: title_from_ref 로 검색 → arxiv PDF
    3) CrossRef: title_from_ref 로 DOI 검색
       → Unpaywall: DOI로 OA PDF URL 검색
    4) (옵션) Semantic Scholar / OpenAlex로 추가 메타데이터
    - 제목은 'title_from_ref'를 우선으로 유지
    """
    result = dict(ref)

    # 기본 필드 안전하게 세팅
    result.setdefault("pdf_url", None)
    result.setdefault("source", "raw")
    result.setdefault("abstract", None)
    result.setdefault("doi", ref.get("doi"))

    # 제목: ref에서 뽑은 title_from_ref가 최우선
    result["title"] = ref.get("title_from_ref")

    raw_text = result["raw"]

    # --------------------------------------------------------
    # 1) 레퍼런스 내부 arXiv ID 직접 추출
    # --------------------------------------------------------
    arxiv_id = extract_arxiv_id(raw_text)
    if arxiv_id:
        print(f"  [Strategy: ArXiv ID in ref] Found ID: {arxiv_id}")

        direct_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        result["pdf_url"] = direct_pdf_url
        result["source"] = "arxiv_direct"

        # 제목이 없으면, arXiv API로 제목/초록만 보충해 줄 수 있음
        if result.get("title") is None:
            found = search_semantic_scholar(f"ARXIV:{arxiv_id}") or search_openalex(arxiv_id)
            if found:
                if found.get("title"):
                    result["title"] = found["title"]
                if found.get("abstract"):
                    result["abstract"] = found["abstract"]
                if found.get("doi") and not result.get("doi"):
                    result["doi"] = found["doi"]

        return result

    # --------------------------------------------------------
    # 검색에 사용할 쿼리 (title_from_ref -> fallback: clean_query)
    # --------------------------------------------------------
    search_query = ref.get("title_from_ref")
    if not search_query:
        search_query = clean_query_aggressive(raw_text)

    print(f"  [Search Query] {search_query[:80]}...")

    if len(search_query.strip()) < 5:
        # 너무 짧으면 그냥 리턴
        return result

    # --------------------------------------------------------
    # 2) arXiv API로 제목 검색
    # --------------------------------------------------------
    print("  [Strategy: arXiv API by title]")
    arxiv_found = search_arxiv_by_title(search_query)
    if arxiv_found and arxiv_found.get("pdf_url"):
        print(f"    -> Found arXiv: {arxiv_found.get('arxiv_id')}")
        result["pdf_url"] = arxiv_found["pdf_url"]
        result["source"] = "arxiv_api"

        # 제목/초록/연도 보정 (단, title_from_ref 있으면 그대로 유지)
        if result.get("title") is None and arxiv_found.get("title"):
            result["title"] = arxiv_found["title"]
        if arxiv_found.get("abstract"):
            result["abstract"] = arxiv_found["abstract"]
        if arxiv_found.get("year") and not result.get("year"):
            result["year"] = str(arxiv_found["year"])

        return result

    # --------------------------------------------------------
    # 3) CrossRef로 DOI 검색 → Unpaywall로 PDF 찾기
    # --------------------------------------------------------
    print("  [Strategy: CrossRef + Unpaywall]")
    crossref_found = search_crossref_by_title(search_query)
    if crossref_found and crossref_found.get("doi"):
        doi = crossref_found["doi"]
        print(f"    -> CrossRef DOI: {doi}")
        result["doi"] = doi
        if crossref_found.get("year") and not result.get("year"):
            result["year"] = str(crossref_found["year"])

        # Unpaywall로 OA PDF 찾기
        upw = get_unpaywall_pdf(doi)
        if upw and upw.get("pdf_url"):
            print(f"    -> Unpaywall PDF URL: {upw['pdf_url']}")
            result["pdf_url"] = upw["pdf_url"]
            result["source"] = upw.get("source", "unpaywall")
            return result
     # --- SciTePress DOI → 직접 PDF 시도 ---
    doi = result.get("doi")
    year = result.get("year")
    if doi and year:
        scitepress_code = extract_scitepress_code_from_doi(doi)
        if scitepress_code:
            print(f"  [Strategy: SciTePress direct] DOI={doi}, year={year}, code={scitepress_code}")
            pdf_url = try_scitepress_pdf(doi, year)
            if pdf_url:
                print(f"    -> Found SciTePress PDF: {pdf_url}")
                result["pdf_url"] = pdf_url
                result["source"] = "scitepress_direct"
                return result
    # --------------------------------------------------------
    # 4) Fallback: Semantic Scholar / OpenAlex (메타데이터 보충용)
    #    - PDF URL이 이미 있으면 제목/초록만 보충
    #    - PDF URL이 없으면 혹시라도 얻을 수 있으면 쓰기
    # --------------------------------------------------------
    print("  [Strategy: Fallback S2/OpenAlex]")
    found = search_semantic_scholar(search_query)
    if not found:
        found = search_openalex(search_query)

    if found:
        # 제목은 title_from_ref가 있는 한 덮어쓰지 않음
        if result.get("title") is None and found.get("title"):
            result["title"] = found["title"]

        if found.get("abstract") and not result.get("abstract"):
            result["abstract"] = found["abstract"]

        if found.get("doi") and not result.get("doi"):
            result["doi"] = found["doi"]

        if found.get("pdf_url") and not result.get("pdf_url"):
            result["pdf_url"] = found["pdf_url"]

        if found.get("source"):
            result["source"] = found["source"]

    return result

# ============================================================
# 5. PDF 다운로드
# ============================================================

def download_pdf_file(url: str, title: str) -> Optional[str]:
    if not url:
        return None
        
    try:
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:100] if title else f"paper_{int(time.time())}"
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        if os.path.exists(filepath):
            return filepath

        print(f"  ... Downloading PDF from {url}")
        r = requests.get(url, headers=HEADERS, stream=True, timeout=20)
        
        if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", "").lower():
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return filepath
        else:
            if r.status_code == 200 and r.content.startswith(b"%PDF"):
                 with open(filepath, "wb") as f:
                    f.write(r.content)
                 return filepath
            
    except Exception as e:
        print(f"  [Download Error] {e}")
    
    return None


# ============================================================
# 6. 메인 파이프라인
# ============================================================

def process_references(pdf_path: str):
    print(f"=== Extracting References from {os.path.basename(pdf_path)} ===")
    
    # ★★ 여기서부터가 핵심 변경 부분 ★★
    # 1. 내가 만든 방식으로 Reference 전체를 구조화해서 추출
    parsed_refs = extract_references_structured(pdf_path)
    
    print(f"Found {len(parsed_refs)} potential references.")
    
    results = []
    
    total = len(parsed_refs)
    for i, ref in enumerate(parsed_refs, start=1):
        print(f"\n[{i}/{total}] Processing (index={ref.get('index')})...")
        
        # 2. 메타데이터 및 링크 검색
        enriched = get_metadata_and_link_enhanced(ref)
        
        # 3. PDF 다운로드
        local_pdf_path = None
        if enriched.get("pdf_url"):
            local_pdf_path = download_pdf_file(enriched["pdf_url"], enriched.get("title") or enriched.get("title_from_ref"))
            
        enriched["local_pdf_path"] = local_pdf_path
        status = "DOWNLOADED" if local_pdf_path else ("LINK FOUND" if enriched.get("pdf_url") else "NOT FOUND")
        
        print(f"  -> Title (from ref): {enriched.get('title_from_ref', 'N/A')}")
        print(f"  -> Title (API)    : {enriched.get('title', 'N/A')}")
        print(f"  -> arXiv block    : {enriched.get('arxiv_block', 'N/A')}")
        print(f"  -> Status         : {status}")
        
        results.append(enriched)
        
        time.sleep(1)  # API Rate Limit
    
    return results

def save_failed_references(results: List[Dict], output_txt: str = "failed_references.txt"):
    failed_count = 0
    
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"=== Failed References Report ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
        f.write("Format: [Index] Reference Raw Text\n\n")
        
        for i, item in enumerate(results):
            if not item.get("local_pdf_path"):
                failed_count += 1
                clean_raw = item['raw'].replace('\n', ' ').strip()
                f.write(f"[{i+1}] {clean_raw}\n")
    
    print(f"\n[INFO] 총 {len(results)}개 중 {failed_count}개를 다운로드하지 못했습니다.")
    print(f"[INFO] 실패 목록 저장 완료: {output_txt}")


# ============================================================
# 실행 예시
# ============================================================

if __name__ == "__main__":
    input_pdf = r"C:\Users\eugen\Downloads\hogan-et-al-2024-human-foot-force-suggests-different-balance-control-between-younger-and-older-adults.pdf"
    
    if os.path.exists(input_pdf):
        final_data = process_references(input_pdf)
        
        with open("reference_results.json", "w", encoding="utf-8") as f:
            json.dump(DOWNLOAD_DIR, f, indent=4, ensure_ascii=False)
            print("\nResult saved to reference_results.json")
        
        save_failed_references(DOWNLOAD_DIR, "failed_references.txt")
    else:
        print(f"File not found: {input_pdf}")

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Iterable
from urllib.parse import urlparse
import re, json

class URLCategory(str, Enum):
    MODEL = "model"
    DATASET = "dataset"
    CODE = "code"
    UNKNOWN = "unknown"
    INVALID = "invalid"

@dataclass
class URLInfo:
    raw_url: str
    category: URLCategory
    valid: bool
    normalized_url: str = ""
    handler: Optional[str] = None
    details: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class URLGroupResult:
    line_no: int
    raw_line: str
    groups: Dict[str, List[str]]
    items: List[URLInfo]
    errors: List[str] = field(default_factory=list)

def _is_valid_http_url(url: str) -> bool:
    try:
        p = urlparse(url.strip())
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False

class GitHubHandler:
    REPO_RE = re.compile(r"^https?://github\\.com/(?P<owner>[^/]+)/(?P<repo>[^/#?]+)")
    def can_handle(self, url: str) -> bool:
        return urlparse(url).netloc.lower() == "github.com"
    def classify(self, url: str) -> URLCategory:
        return URLCategory.CODE
    def normalize(self, url: str) -> str:
        m = self.REPO_RE.match(url)
        if m:
            return f"https://github.com/{m.group('owner')}/{m.group('repo')}"
        return url
    def extract_details(self, url: str) -> dict:
        m = self.REPO_RE.match(url)
        if not m: return {}
        return {"service": "github", "owner": m.group("owner"), "repo": m.group("repo")}

class HuggingFaceDatasetHandler:
    def can_handle(self, url: str) -> bool:
        netloc = urlparse(url).netloc.lower()
        return netloc.endswith("huggingface.co") and "/datasets/" in urlparse(url).path
    def classify(self, url: str) -> URLCategory:
        return URLCategory.DATASET
    def normalize(self, url: str) -> str:
        parts = [p for p in urlparse(url).path.split("/") if p]
        try:
            idx = parts.index("datasets")
            norm = "/".join(parts[:idx+3])
            return f"https://huggingface.co/{norm}"
        except ValueError:
            return url
    def extract_details(self, url: str) -> dict:
        parts = [p for p in urlparse(url).path.split("/") if p]
        if "datasets" in parts:
            i = parts.index("datasets")
            org = parts[i+1] if len(parts) > i+1 else ""
            name = parts[i+2] if len(parts) > i+2 else ""
            return {"service": "huggingface", "kind": "dataset", "org": org, "name": name}
        return {}

class HuggingFaceModelHandler:
    def can_handle(self, url: str) -> bool:
        pr = urlparse(url)
        if pr.netloc.lower().endswith("huggingface.co"):
            path = pr.path.strip("/")
            if not path or path.startswith(("datasets/", "spaces", "docs")):
                return False
            return True
        return False
    def classify(self, url: str) -> URLCategory:
        return URLCategory.MODEL
    def normalize(self, url: str) -> str:
        parts = [p for p in urlparse(url).path.split("/") if p]
        if len(parts) >= 2:
            return f"https://huggingface.co/{parts[0]}/{parts[1]}"
        return url
    def extract_details(self, url: str) -> dict:
        parts = [p for p in urlparse(url).path.split("/") if p]
        org = parts[0] if len(parts) >= 1 else ""
        name = parts[1] if len(parts) >= 2 else ""
        return {"service": "huggingface", "kind": "model", "org": org, "name": name}

def classify_with_handlers(url: str) -> URLInfo:
    if not _is_valid_http_url(url):
        return URLInfo(raw_url=url, category=URLCategory.INVALID, valid=False, normalized_url=url, error="invalid_url")
    handlers = [HuggingFaceDatasetHandler(), HuggingFaceModelHandler(), GitHubHandler()]
    for h in handlers:
        if h.can_handle(url):
            return URLInfo(
                raw_url=url,
                category=h.classify(url),
                valid=True,
                normalized_url=h.normalize(url),
                handler=h.__class__.__name__,
                details=h.extract_details(url)
            )
    return URLInfo(raw_url=url, category=URLCategory.UNKNOWN, valid=True, normalized_url=url, details={})

class URLProcessor:
    def _iter_url_tokens(self, line: str) -> Iterable[str]:
        for token in re.split(r"[\\s,]+", line.strip()):
            if token and not token.startswith("#"):
                yield token
    def process_file(self, path: str) -> List[URLGroupResult]:
        results: List[URLGroupResult] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, raw in enumerate(f, start=1):
                raw_line = raw.strip()
                if not raw_line or raw_line.startswith("#"): continue
                items = [classify_with_handlers(tok) for tok in self._iter_url_tokens(raw_line)]
                groups = {
                    "model_urls": [x.normalized_url for x in items if x.valid and x.category == URLCategory.MODEL],
                    "dataset_urls": [x.normalized_url for x in items if x.valid and x.category == URLCategory.DATASET],
                    "code_urls": [x.normalized_url for x in items if x.valid and x.category == URLCategory.CODE],
                }
                errors = [x.error for x in items if x.error]
                results.append(URLGroupResult(i, raw_line, groups, items, errors))
        return results
    def to_json(self, results: List[URLGroupResult]) -> str:
        serializable = []
        for r in results:
            serializable.append({
                "line_no": r.line_no,
                "raw_line": r.raw_line,
                "groups": r.groups,
                "items": [asdict(x) for x in r.items],
                "errors": r.errors,
            })
        return json.dumps(serializable, indent=2)

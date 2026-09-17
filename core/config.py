import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

try:
    import torch
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)
    USE_GPU = torch.cuda.is_available()
except Exception:
    torch = None
    USE_GPU = False

# Resilient model cache directories (defaults to .cache in project root)
DEFAULT_CACHE_BASE = os.path.join(BASE_DIR, ".cache")
CACHE_DIR = os.environ.get("HF_HOME") or os.path.join(DEFAULT_CACHE_BASE, "huggingface")
TORCH_CACHE = os.environ.get("TORCH_HOME") or os.path.join(DEFAULT_CACHE_BASE, "torch")
EASYOCR_CACHE = os.environ.get("EASYOCR_MODULE_PATH") or os.path.join(DEFAULT_CACHE_BASE, "easyocr")

# Ensure directories exist
for path in [UPLOAD_DIR, CACHE_DIR, TORCH_CACHE, EASYOCR_CACHE]:
    os.makedirs(path, exist_ok=True)

# Environment variables setup for cache & CPU thread tuning
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = TORCH_CACHE
os.environ["EASYOCR_MODULE_PATH"] = EASYOCR_CACHE
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Image Preprocessing & OCR Constraints
MAX_IMAGE_LONG_EDGE = 1600
MIN_SHARPNESS_LAPLACIAN = 80.0

# VLM Inference settings
VLM_MODEL_ID = "microsoft/Florence-2-base"
VLM_TARGET_DIM = 768
VLM_MAX_NEW_TOKENS = 128

# Compliance & GTIN Matching
COMPLIANCE_PASS_THRESHOLD = 85.0

# Database
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'praman.db')}")

# Gemini LLM Integration
# Reads GEMINI_API_KEY from environment or .env securely without logging
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("API_KEY")
)

# JWT Auth & Security
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

SECRET_KEY = os.environ.get("PRAMAN_SECRET_KEY", "praman-dev-secret-CHANGE-IN-PROD")
if ENVIRONMENT == "production" and SECRET_KEY == "praman-dev-secret-CHANGE-IN-PROD":
    raise ValueError("PRAMAN_SECRET_KEY must be configured with a secure random key when ENVIRONMENT=production")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# CORS — Expo Metro, Expo Web, React Native dev, and web scanner
_raw_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:8081,http://localhost:19006,http://localhost:3000,http://127.0.0.1:8000,http://localhost:8000"
)
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Quality & Processing thresholds from Codebase 2 spec
GLARE_THRESHOLD = 0.18          # reject images where >18% pixels are overexposed
OCR_CONFIDENCE_THRESHOLD = 0.75 # gate for cascade: below this triggers Tier 2+3
IOU_MERGE_THRESHOLD = 0.45      # spatial merger: boxes with IoU >= this are merged

# Authenticity score bands (integer 0–100)
AUTHENTICITY_PASS_MIN = 85
AUTHENTICITY_WARN_MIN = 50

# Pagination
SCAN_HISTORY_PAGE_SIZE = 20


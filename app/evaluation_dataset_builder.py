from __future__ import annotations

import re
import unicodedata
from pathlib import Path


PROVIDER_KEYWORDS = {
    "Microsoft": ["microsoft", "power bi", "office", "windows", "school agreement"],
    "ESET": ["eset"],
    "Sophos": ["sophos"],
    "Fortinet": ["fortinet", "fortigate", "fortimail"],
    "Google": ["google", "workspace", "gmail"],
    "Adobe": ["adobe", "acrobat", "creative cloud"],
    "Zoom": ["zoom"],
    "ForeFlight": ["foreflight", "jeppesen"],
    "Sectigo": ["sectigo"],
}


def infer_expert_label(text: str, document_name: str = "") -> dict:
    normalized = _normalize(" ".join([document_name, text]))
    category = "Servicios profesionales TI"
    solution = "Servicio profesional TI"
    has_hosting = _has_any(
        normalized,
        ["hosting", "cpanel", "vps", "pagina web", "sitio web", "dominio gob.ec", "dominio institucional"],
    )
    has_ssl = _has_any(normalized, ["certificado ssl", "ssl", "tls", "https", "dominio unico", "wildcard"])

    if _has_any(normalized, ["power bi", "business intelligence", "inteligencia de negocios", "licenciamiento power bi"]):
        category = "Business Intelligence"
        solution = "Microsoft Power BI Pro"
    elif _has_any(normalized, ["foreflight", "jeppesen", "cartas ifr"]):
        category = "Licenciamiento de software"
        solution = "ForeFlight Performance Plus + Jeppesen IFR Charts"
    elif _has_any(normalized, ["microsoft 365", "office 365", "school agreement", "windows education", "licencias microsoft"]):
        category = "Licenciamiento de software"
        solution = "Microsoft 365 Business Standard"
    elif _has_any(normalized[:3000], ["software antivirus", "licencias antivirus", "solucion antivirus", "antivirus para"]):
        category = "Antivirus / EDR / XDR"
        if "eset" in normalized:
            solution = "ESET PROTECT"
        elif "sophos" in normalized or "xdr" in normalized:
            solution = "Sophos Intercept X Advanced with XDR"
        else:
            solution = "Solucion antivirus / EDR administrada"
    elif _has_any(normalized, ["firewall", "seguridad perimetral", "fortinet", "fortigate"]):
        category = "Firewall"
        solution = "Fortinet FortiGate"
    elif _has_any(normalized, ["antivirus", "endpoint", "edr", "xdr", "ransomware", "eset", "sophos", "malware"]):
        category = "Antivirus / EDR / XDR"
        if "eset" in normalized:
            solution = "ESET PROTECT"
        elif "sophos" in normalized or "xdr" in normalized:
            solution = "Sophos Intercept X Advanced with XDR"
        else:
            solution = "Solucion antivirus / EDR administrada"
    elif has_ssl and not has_hosting:
        category = "Hosting / cPanel / VPS"
        solution = "Certificados SSL"
    elif has_hosting:
        category = "Hosting / cPanel / VPS"
        solution = "Hosting cPanel"
    elif _has_any(normalized, ["microsoft 365", "office", "school agreement", "windows education", "licencias microsoft"]):
        category = "Licenciamiento de software"
        solution = "Microsoft 365 Business Standard"
    elif _has_any(normalized, ["linux", "red hat", "suse", "ubuntu"]):
        category = "Licenciamiento de software"
        solution = "Suscripción de licencias Linux"
    elif _has_any(normalized, ["foreflight", "jeppesen", "cartas ifr"]):
        category = "Licenciamiento de software"
        solution = "ForeFlight Performance Plus + Jeppesen IFR Charts"
    elif _has_any(normalized, ["videoconferencia", "telepresenciales", "salas virtuales", "reuniones virtuales"]):
        category = "Colaboración y videoconferencia"
        solution = "Zoom Workplace"
    elif _has_any(normalized, ["correo masivo", "email marketing", "envio masivo de correos", "contactos"]):
        category = "Cloud / SaaS"
        solution = "Plataforma SaaS de correo masivo"
    elif _has_any(normalized, ["plataforma de inteligencia artificial", "chatgpt", "inteligencia artificial profesional"]):
        category = "Cloud / SaaS"
        solution = "Plataforma de inteligencia artificial profesional"
    elif _has_any(normalized, ["base de datos cientifica", "biblioteca", "analisis e inteligencia artificial"]):
        category = "Herramientas educativas"
        solution = "Base de datos científica internacional con herramientas de análisis e IA"

    requirements = _extract_expected_requirements(text, category, solution)
    return {
        "expected_category": category,
        "expected_solution": solution,
        "expected_requirements": requirements,
    }


def build_case_from_text(
    text: str,
    document_name: str,
    source_path: str,
    case_id: str,
    split: str,
    document_type: str,
) -> dict:
    label = infer_expert_label(text, document_name)
    provider = _provider_bucket(" ".join([document_name, text]))
    return {
        "id": case_id,
        "document_name": document_name,
        "text": text.strip(),
        "expected_category": label["expected_category"],
        "expected_solution": label["expected_solution"],
        "expected_requirements": label["expected_requirements"],
        "groups": {
            "category": _category_group(label["expected_category"]),
            "provider": provider,
            "document_quality": _document_quality(text),
            "document_type": document_type,
            "modality": _modality_bucket(text),
            "length": _length_bucket(text),
            "source_month": "2026-07",
        },
        "split": split,
        "source_path": source_path,
        "labeling_method": "expert_curated_rules_from_july_tdr_context",
    }


def extract_pdf_text(path: str | Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF no esta instalado. Ejecute: pip install -r requirements.txt") from exc

    pdf_path = Path(path)
    document = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in document)
    return _clean_text(text)


def is_candidate_tdr_text(text: str, name: str) -> bool:
    normalized = _normalize(" ".join([name, text[:6000]]))
    if _has_any(normalized, ["analisis comparativo", "proyecto integrador", "evaluacion y optimizacion"]):
        return False
    if _has_any(normalized, ["factura", "comprobante de retencion", "cedula"]):
        return False
    signals = [
        "terminos de referencia",
        "objeto de contratacion",
        "identificacion del objeto",
        "especificaciones tecnicas",
        "formulacion de la necesidad",
        "area requirente",
        "orden de compra",
        "presupuesto pr",
    ]
    return sum(1 for signal in signals if signal in normalized) >= 1


def _extract_expected_requirements(text: str, category: str, solution: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in re.split(r"[\n.;•]+", text):
        line = re.sub(r"\s+", " ", raw_line).strip(" -:•\t")
        if not 18 <= len(line) <= 260:
            continue
        normalized = _normalize(line)
        if _has_any(
            normalized,
            [
                "debe",
                "incluye",
                "incluir",
                "soporte",
                "licencia",
                "suscripcion",
                "certificado",
                "hosting",
                "dominio",
                "backup",
                "firewall",
                "antivirus",
                "endpoint",
                "dashboard",
                "usuarios",
                "capacitacion",
                "implementacion",
                "configuracion",
                "administracion",
                "vigencia",
                "garantia",
            ],
        ):
            candidates.append(line)

    fallback = _fallback_requirements(category, solution)
    merged = []
    seen = set()
    for item in candidates + fallback:
        key = _normalize(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= 8:
            break
    return merged or fallback[:5]


def _fallback_requirements(category: str, solution: str) -> list[str]:
    if category == "Business Intelligence":
        return [
            "licencias para creación de reportes y dashboards",
            "publicación y uso seguro de reportes",
            "integración con fuentes de datos institucionales",
        ]
    if category == "Antivirus / EDR / XDR":
        return [
            "protección antivirus para endpoints y servidores",
            "administración centralizada de la solución",
            "soporte técnico durante la vigencia del servicio",
        ]
    if category == "Firewall":
        return [
            "seguridad perimetral con firewall de próxima generación",
            "funcionalidades de VPN, IPS y filtrado de contenido",
            "soporte técnico y licenciamiento vigente",
        ]
    if solution == "Certificados SSL":
        return [
            "certificado SSL/TLS para servicios web institucionales",
            "instalación y configuración del certificado",
            "soporte técnico especializado",
        ]
    if category == "Hosting / cPanel / VPS":
        return [
            "hosting o alojamiento web para sitio institucional",
            "dominio y certificado SSL",
            "soporte técnico y respaldos de información",
        ]
    if category == "Colaboración y videoconferencia":
        return [
            "licencias para reuniones virtuales",
            "acceso seguro a salas de videoconferencia",
            "soporte técnico durante la vigencia",
        ]
    return [
        f"contratación de {solution}",
        "soporte técnico durante la vigencia del servicio",
        "entrega de evidencias de activación o implementación",
    ]


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9ñ\s/._-]", " ", value)


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _has_any(normalized_text: str, terms: list[str]) -> bool:
    return any(term in normalized_text for term in terms)


def _provider_bucket(text: str) -> str:
    normalized = _normalize(text)
    for provider, terms in PROVIDER_KEYWORDS.items():
        if any(term in normalized for term in terms):
            return provider
    return "genérico"


def _category_group(category: str) -> str:
    if category in {"Hosting / cPanel / VPS"}:
        return "Hosting / certificados"
    if category in {"Cloud / SaaS", "Colaboración y videoconferencia"}:
        return "SaaS / colaboración"
    if category in {"Antivirus / EDR / XDR", "Firewall"}:
        return "Ciberseguridad"
    return category


def _document_quality(text: str) -> str:
    words = len(text.split())
    if words < 80:
        return "extracción limitada"
    if words > 2500:
        return "largo"
    return "claro"


def _modality_bucket(text: str) -> str:
    normalized = _normalize(text)
    if _has_any(normalized, ["saas", "nube", "cloud", "plataforma"]):
        return "SaaS"
    if _has_any(normalized, ["hardware", "appliance", "equipo"]):
        return "hardware"
    if _has_any(normalized, ["licencia", "suscripcion", "suscripción"]):
        return "licencia/suscripción"
    return "servicio"


def _length_bucket(text: str) -> str:
    words = len(text.split())
    if words < 500:
        return "corto"
    if words < 1800:
        return "medio"
    return "largo"

from __future__ import annotations

from app.schemas import ClassificationResult
from app.text_cleaner import normalize_for_matching


CATEGORIES: dict[str, list[str]] = {
    "Licenciamiento de software": [
        "licencia",
        "licencias",
        "suscripcion",
        "microsoft 365",
        "office",
        "adobe",
        "acrobat",
        "usuarios",
    ],
    "Ciberseguridad": [
        "ciberseguridad",
        "seguridad informatica",
        "amenazas",
        "vulnerabilidades",
        "incidentes",
        "proteccion",
    ],
    "Antivirus / EDR / XDR": [
        "antivirus",
        "edr",
        "xdr",
        "endpoint",
        "estaciones de trabajo",
        "malware",
        "ransomware",
    ],
    "Firewall": ["firewall", "fortigate", "utm", "vpn", "ips", "filtrado web", "perimetral"],
    "Antispam / correo seguro": [
        "antispam",
        "correo seguro",
        "phishing",
        "mail gateway",
        "fortimail",
        "spam",
    ],
    "Cloud / SaaS": ["saas", "nube", "cloud", "tenant", "servicio cloud", "plataforma web"],
    "Hosting / cPanel / VPS": ["hosting", "cpanel", "vps", "servidor virtual", "dominio", "ssl"],
    "Backup y recuperación": [
        "backup",
        "respaldo",
        "recuperacion",
        "retencion",
        "copias",
        "restauracion",
    ],
    "Colaboración y videoconferencia": [
        "videoconferencia",
        "teams",
        "zoom",
        "reuniones",
        "chat",
        "colaboracion",
    ],
    "Business Intelligence": [
        "business intelligence",
        "bi",
        "dashboard",
        "power bi",
        "indicadores",
        "reportes",
        "analitica",
    ],
    "Herramientas educativas": [
        "educativa",
        "lms",
        "moodle",
        "estudiantes",
        "cursos",
        "matriculas",
        "aula virtual",
    ],
    "Hardware": ["hardware", "equipo", "servidor fisico", "switch", "router", "computador"],
    "Servicios profesionales TI": [
        "consultoria",
        "implementacion",
        "soporte tecnico",
        "capacitacion",
        "mesa de ayuda",
        "servicios profesionales",
    ],
}


def classify_requirement(text: str) -> ClassificationResult:
    normalized = normalize_for_matching(text)
    scored: list[tuple[float, str, list[str]]] = []

    for category, keywords in CATEGORIES.items():
        evidence = [keyword for keyword in keywords if keyword in normalized]
        score = float(len(evidence))
        if category == "Antivirus / EDR / XDR" and ("edr" in normalized or "xdr" in normalized):
            score += 2.0
        if category == "Business Intelligence" and "power bi" in normalized:
            score += 2.0
        if category == "Licenciamiento de software" and "licencia" in normalized:
            score += 1.5
        if category == "Cloud / SaaS" and ("saas" in normalized or "nube" in normalized):
            score += 1.0
        if score:
            scored.append((score, category, evidence))

    if not scored:
        return ClassificationResult("Otro / no identificado", 0.0, [])

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_category, evidence = scored[0]
    total = sum(item[0] for item in scored)
    confidence = min(0.98, best_score / total if total else 0.0)
    return ClassificationResult(best_category, round(confidence, 2), evidence[:6])

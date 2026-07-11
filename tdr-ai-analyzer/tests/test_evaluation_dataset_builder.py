from app.evaluation_dataset_builder import build_case_from_text, infer_expert_label


def test_infer_expert_label_identifies_power_bi_tdr():
    text = """
    TERMINOS DE REFERENCIA
    OBJETO DE CONTRATACION: ADQUISICION LICENCIAMIENTO POWER BI.
    Debe permitir creación de reportes y dashboards interactivos.
    Debe integrarse con archivos Excel y bases de datos.
    """

    label = infer_expert_label(text, "power-bi.pdf")

    assert label["expected_category"] == "Business Intelligence"
    assert label["expected_solution"] == "Microsoft Power BI Pro"
    assert "dashboard" in " ".join(label["expected_requirements"]).lower()


def test_infer_expert_label_identifies_ssl_certificate_tdr():
    text = """
    Terminos de referencia para contratar certificados SSL.
    Objeto de contratacion: certificado SSL dominio unico para plataforma web.
    Debe incluir soporte técnico para instalación y configuración.
    """

    label = infer_expert_label(text, "ssl.pdf")

    assert label["expected_category"] == "Hosting / cPanel / VPS"
    assert label["expected_solution"] == "Certificados SSL"
    assert any("ssl" in requirement.lower() for requirement in label["expected_requirements"])


def test_infer_expert_label_prioritizes_firewall_over_generic_virus_terms():
    text = """
    TERMINOS DE REFERENCIA
    CONTRATACION DEL SERVICIO DE SEGURIDAD PERIMETRAL DE FIREWALL.
    Se requiere proteccion ante virus, malware y phishing.
    El appliance debe incluir VPN, IPS, filtrado URL y control de aplicaciones.
    """

    label = infer_expert_label(text, "firewall.pdf")

    assert label["expected_category"] == "Firewall"
    assert label["expected_solution"] == "Fortinet FortiGate"


def test_infer_expert_label_does_not_use_generic_indicators_as_power_bi_signal():
    text = """
    TERMINOS DE REFERENCIA
    OBJETO DE CONTRATACION: ADQUISICION, INSTALACION Y DESPLIEGUE DE SOFTWARE ANTIVIRUS.
    El proveedor debe entregar informes, indicadores del servicio y consola centralizada.
    Debe proteger estaciones, servidores y correo institucional contra malware.
    """

    label = infer_expert_label(text, "5847685.pdf")

    assert label["expected_category"] == "Antivirus / EDR / XDR"
    assert label["expected_solution"] == "Solucion antivirus / EDR administrada"


def test_infer_expert_label_does_not_treat_vpn_ips_features_as_firewall_by_themselves():
    text = """
    TERMINOS DE REFERENCIA
    OBJETO DE CONTRATACION: ADQUISICION, INSTALACION Y DESPLIEGUE DE SOFTWARE ANTIVIRUS.
    La solucion debe incluir proteccion web, filtrado URL, IPS, VPN segura y defensa contra malware.
    Debe tener consola centralizada para estaciones y servidores.
    """

    label = infer_expert_label(text, "5847685.pdf")

    assert label["expected_category"] == "Antivirus / EDR / XDR"


def test_infer_expert_label_prioritizes_antivirus_object_over_firewall_feature():
    text = """
    PROYECTO TERMINOS DE REFERENCIA
    OBJETO DE LA CONTRATACION
    ADQUISICION, INSTALACION Y DESPLIEGUE DE SOFTWARE ANTIVIRUS PARA LOS EQUIPOS PC
    Y SERVIDORES ACTIVOS.
    La solucion debe incluir firewall inteligente, control de trafico y monitoreo de aplicaciones.
    """

    label = infer_expert_label(text, "5847685.pdf")

    assert label["expected_category"] == "Antivirus / EDR / XDR"


def test_infer_expert_label_prioritizes_microsoft_school_agreement_over_endpoint_terms():
    text = """
    TERMINOS DE REFERENCIA
    OBJETO DE CONTRATACION: ADQUISICION DE SUSCRIPCION PARA LICENCIAS MICROSOFT SCHOOL AGREEMENT
    Y SEGURIDAD ENDPOINT.
    Incluye licencias Microsoft 365, Office y Windows Education para usuarios institucionales.
    """

    label = infer_expert_label(text, "7750087.pdf")

    assert label["expected_category"] == "Licenciamiento de software"
    assert label["expected_solution"] == "Microsoft 365 Business Standard"


def test_infer_expert_label_prioritizes_hosting_when_ssl_is_secondary():
    text = """
    TERMINOS DE REFERENCIA
    OBJETO DE CONTRATACION: REACTIVACION, REDISENO, HOSTING, DOMINIO Y ADMINISTRACION
    DE LA PAGINA WEB INSTITUCIONAL.
    El servicio incluye certificado SSL, respaldos, soporte tecnico y administracion cPanel.
    """

    label = infer_expert_label(text, "7745298.pdf")

    assert label["expected_category"] == "Hosting / cPanel / VPS"
    assert label["expected_solution"] == "Hosting cPanel"


def test_build_case_from_text_adds_metadata_groups_and_split():
    text = """
    Términos de referencia para contratación de licencias antivirus ESET.
    Debe incluir consola web de administración centralizada.
    Debe proteger endpoints y servidores.
    """

    case = build_case_from_text(
        text=text,
        document_name="antivirus.pdf",
        source_path="C:/tmp/antivirus.pdf",
        case_id="case-001",
        split="validation",
        document_type="real_tdr",
    )

    assert case["id"] == "case-001"
    assert case["document_name"] == "antivirus.pdf"
    assert case["split"] == "validation"
    assert case["expected_category"] == "Antivirus / EDR / XDR"
    assert case["groups"]["document_type"] == "real_tdr"
    assert case["groups"]["provider"] == "ESET"
    assert case["source_path"] == "C:/tmp/antivirus.pdf"

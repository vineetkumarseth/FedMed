"""Development PKI for gRPC/TLS: a private CA plus a server certificate.

For production use certificates issued by your organisation's CA instead.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
from pathlib import Path
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def _name(cn: str) -> x509.Name:
    return x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedMed Consortium"),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])


def _pem_key(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def generate_dev_certs(cert_dir: str | Path, hosts: Tuple[str, ...] = ("localhost", "127.0.0.1")) -> Path:
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = (x509.CertificateBuilder()
               .subject_name(_name("FedMed Dev CA")).issuer_name(_name("FedMed Dev CA"))
               .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
               .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=365))
               .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
               .sign(ca_key, hashes.SHA256()))

    srv_key = ec.generate_private_key(ec.SECP256R1())
    sans = []
    for h in hosts:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    srv_cert = (x509.CertificateBuilder()
                .subject_name(_name("fedmed-server")).issuer_name(ca_cert.subject)
                .public_key(srv_key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=365))
                .add_extension(x509.SubjectAlternativeName(sans), critical=False)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .sign(ca_key, hashes.SHA256()))

    (cert_dir / "ca.crt").write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    (cert_dir / "server.crt").write_bytes(srv_cert.public_bytes(serialization.Encoding.PEM))
    (cert_dir / "server.key").write_bytes(_pem_key(srv_key))
    (cert_dir / "server.key").chmod(0o600)
    return cert_dir


def load_server_certs(cert_dir: str | Path) -> Tuple[bytes, bytes, bytes]:
    d = Path(cert_dir)
    return (d / "ca.crt").read_bytes(), (d / "server.crt").read_bytes(), (d / "server.key").read_bytes()


def load_ca(cert_dir: str | Path) -> bytes:
    return (Path(cert_dir) / "ca.crt").read_bytes()

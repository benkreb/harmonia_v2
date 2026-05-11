"""Generate a local self-signed CA plus Flower server/client mTLS certs."""

from __future__ import annotations

import ipaddress
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


OUT = Path(__file__).resolve().parent / "generated"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ca_key = _key()
    ca_cert = _cert("harmonia-fl-ca", ca_key, ca_key, None, is_ca=True)
    _write("ca", ca_key, ca_cert)

    server_key = _key()
    server_cert = _cert(
        "fl-server",
        server_key,
        ca_key,
        ca_cert,
        alt_names=[x509.DNSName("fl-server"), x509.IPAddress(ipaddress.ip_address("10.10.0.1"))],
    )
    _write("server", server_key, server_cert)

    for client in ["client_1", "client_2"]:
        key = _key()
        cert = _cert(client, key, ca_key, ca_cert)
        _write(client, key, cert)


def _key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _cert(common_name, key, issuer_key, issuer_cert, alt_names=None, is_ca=False):
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    issuer = issuer_cert.subject if issuer_cert else subject
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=5))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), critical=True)
    )
    if alt_names:
        builder = builder.add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
    return builder.sign(private_key=issuer_key, algorithm=hashes.SHA256())


def _write(name, key, cert) -> None:
    (OUT / f"{name}.key").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    (OUT / f"{name}.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))


if __name__ == "__main__":
    main()


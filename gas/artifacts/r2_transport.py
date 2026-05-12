from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


class R2TransportError(Exception):
    pass


def join_key(prefix: str, key: str) -> str:
    prefix = (prefix or "").strip("/")
    key = (key or "").strip("/")
    if not prefix:
        return key
    if not key:
        return prefix
    if key == prefix or key.startswith(f"{prefix}/"):
        return key
    return f"{prefix}/{key}"


class ArtifactR2Transport:
    """Small S3-compatible transport used for DPS artifact manifests/files."""

    def read_url(self, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            return Path(parsed.path).read_bytes()
        if parsed.scheme in ("http", "https"):
            with urlopen(url, timeout=30) as response:
                return response.read()
        return Path(url).read_bytes()

    def read_object(
        self,
        *,
        endpoint_url: str = "",
        bucket: str,
        key: str,
        region: str = "auto",
        access_key_id: str = "",
        secret_access_key: str = "",
        session_token: str = "",
    ) -> bytes:
        endpoint_url = endpoint_url or ""
        if endpoint_url.startswith("file://"):
            root = Path(urlparse(endpoint_url).path)
            return (root / bucket / key).read_bytes()

        if self._should_use_s3(endpoint_url, access_key_id, secret_access_key):
            client = self._s3_client(
                endpoint_url=endpoint_url,
                region=region,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                session_token=session_token,
            )
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()

        if endpoint_url.startswith(("http://", "https://")):
            return self.read_url(f"{endpoint_url.rstrip('/')}/{bucket}/{key}")

        return (Path(bucket) / key).read_bytes()

    def write_object(
        self,
        *,
        endpoint_url: str = "",
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        region: str = "auto",
        access_key_id: str = "",
        secret_access_key: str = "",
        session_token: str = "",
    ) -> None:
        endpoint_url = endpoint_url or ""
        if endpoint_url.startswith("file://"):
            root = Path(urlparse(endpoint_url).path)
            path = root / bucket / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return

        if self._should_use_s3(endpoint_url, access_key_id, secret_access_key):
            client = self._s3_client(
                endpoint_url=endpoint_url,
                region=region,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                session_token=session_token,
            )
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            return

        path = Path(bucket) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def _should_use_s3(
        self,
        endpoint_url: str,
        access_key_id: str = "",
        secret_access_key: str = "",
    ) -> bool:
        return bool(endpoint_url and access_key_id and secret_access_key)

    def _s3_client(
        self,
        *,
        endpoint_url: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        session_token: str = "",
    ):
        try:
            import boto3
        except ImportError as e:
            raise R2TransportError(
                "boto3 is required for authenticated R2 transport"
            ) from e

        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region or "auto",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token or None,
        )

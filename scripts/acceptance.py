from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict | None = None,
    expected_status: int = 200,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = json.loads(exc.read())
    if status != expected_status:
        raise AssertionError(
            f"{method} {path}: esperado {expected_status}, recibido {status}: {body}"
        )
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Aceptacion del servicio NovaMercado")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--reentrenar", action="store_true")
    parser.add_argument("--timeout-retrain", type=int, default=600)
    args = parser.parse_args()

    examples = args.data_dir / "ejemplos"
    valid_prediction = json.loads((examples / "prediccion_valida.json").read_text())
    valid_observation = json.loads((examples / "observacion_valida.json").read_text())
    duplicate_family = json.loads(
        (examples / "errores" / "E06_familia_repetida.json").read_text()
    )

    health = request_json(args.base_url, "GET", "/health")
    assert health["estado"] == "ok", health
    print("[ok] salud completa")

    model = request_json(args.base_url, "GET", "/modelo")
    assert set(model) == {"nombre", "alias", "version"}
    print(f"[ok] modelo vigente {model['nombre']}@{model['alias']} version {model['version']}")

    forecast = request_json(args.base_url, "POST", "/v1/prediccion", valid_prediction)
    assert forecast["version_modelo"] == model["version"]
    assert len(forecast["predicciones"]) == len(valid_prediction["familias"])
    assert all(item["cantidad_sugerida"] >= 0 for item in forecast["predicciones"])
    print("[ok] contrato de pronostico")

    rejected = request_json(
        args.base_url,
        "POST",
        "/v1/prediccion",
        duplicate_family,
        expected_status=422,
    )
    assert isinstance(rejected["detail"], str)
    print("[ok] rechazo 422 con detalle textual")

    first = request_json(args.base_url, "POST", "/v1/observaciones", valid_observation)
    second = request_json(args.base_url, "POST", "/v1/observaciones", valid_observation)
    assert first["registros_guardados"] >= 0
    assert second["registros_guardados"] == 0
    print("[ok] observaciones idempotentes")

    if args.reentrenar:
        started = request_json(
            args.base_url,
            "POST",
            "/admin/reentrenar",
            expected_status=202,
        )
        run_id = started["id"]
        deadline = time.monotonic() + args.timeout_retrain
        while time.monotonic() < deadline:
            status = request_json(args.base_url, "GET", f"/admin/reentrenar/{run_id}")
            if status["estado"] in {"exitoso", "fallido"}:
                assert status["estado"] == "exitoso", status
                print(f"[ok] reentrenamiento {run_id}: {status}")
                break
            time.sleep(3)
        else:
            raise AssertionError("El reentrenamiento excedio el timeout")


if __name__ == "__main__":
    main()

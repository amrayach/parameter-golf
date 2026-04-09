#!/usr/bin/env python3
"""Retry allocation of a single 8x H100 SXM RunPod pod until capacity appears.

RunPod identifies the H100 SXM SKU by the GPU type id
`NVIDIA H100 80GB HBM3`. That id is the parity target for this script.

Requires:
  - RUNPOD_API_KEY in the environment

Examples:
  python3 scripts/runpod_retry_h100.py acquire
  python3 scripts/runpod_retry_h100.py acquire --name parameter-golf-07c1-h100 --dry-run
  python3 scripts/runpod_retry_h100.py status <pod-id>
  python3 scripts/runpod_retry_h100.py delete <pod-id>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


API_BASE = "https://rest.runpod.io/v1"
# RunPod's GPU type id for the H100 SXM 80 GB SKU.
DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"
DEFAULT_TEMPLATE_ID = "y5cejece4j"
# `AP-IN-1` appeared in a connector path but is currently rejected by the
# public REST schema. Keep the default set to REST-valid datacenter ids.
DEFAULT_REGIONS: list[str] | None = None


class RunpodApiError(RuntimeError):
    pass


@dataclass
class ApiResponse:
    status: int
    payload: Any


def require_api_key() -> str:
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise SystemExit("RUNPOD_API_KEY is required in the environment.")
    return token


def request_json(
    token: str,
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> ApiResponse:
    url = f"{API_BASE}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body) if body else None
            return ApiResponse(status=resp.status, payload=parsed)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = body or None
        return ApiResponse(status=exc.code, payload=parsed)


def extract_error_message(payload: Any) -> str:
    if payload is None:
        return "empty response"
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if value:
                return str(value)
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            msgs = []
            for item in errors:
                if isinstance(item, dict):
                    msg = item.get("message") or json.dumps(item, sort_keys=True)
                else:
                    msg = str(item)
                msgs.append(msg)
            return "; ".join(msgs)
    return json.dumps(payload, sort_keys=True)


def extract_allowed_datacenter_ids(payload: Any) -> set[str]:
    allowed: set[str] = set()
    if not isinstance(payload, list):
        return allowed
    for item in payload:
        if not isinstance(item, dict):
            continue
        problems = item.get("problems")
        if not isinstance(problems, list):
            continue
        for problem in problems:
            if not isinstance(problem, str):
                continue
            if "/pods/properties/dataCenterIds/items/enum" not in problem:
                continue
            allowed.update(re.findall(r"'([^']+)'", problem))
    return allowed


def list_existing_pods(token: str, name: str) -> list[dict[str, Any]]:
    resp = request_json(token, "GET", "/pods", query={"name": name})
    if resp.status != 200:
        raise RunpodApiError(f"list pods failed: {extract_error_message(resp.payload)}")
    if not isinstance(resp.payload, list):
        return []
    return [
        pod
        for pod in resp.payload
        if isinstance(pod, dict) and str(pod.get("name", "")) == name
    ]


def get_pod(token: str, pod_id: str) -> dict[str, Any] | None:
    resp = request_json(token, "GET", f"/pods/{pod_id}")
    if resp.status == 404:
        return None
    if resp.status != 200:
        raise RunpodApiError(f"get pod failed: {extract_error_message(resp.payload)}")
    if not isinstance(resp.payload, dict):
        raise RunpodApiError("get pod returned a non-object payload")
    return resp.payload


def delete_pod(token: str, pod_id: str) -> None:
    resp = request_json(token, "DELETE", f"/pods/{pod_id}")
    if resp.status not in (200, 202, 204):
        raise RunpodApiError(f"delete pod failed: {extract_error_message(resp.payload)}")


def build_create_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cloudType": "SECURE",
        "computeType": "GPU",
        "name": args.name,
        "gpuTypeIds": [args.gpu_type],
        "gpuCount": args.gpu_count,
        "gpuTypePriority": "availability",
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": args.volume_gb,
        "volumeMountPath": args.volume_mount_path,
        "ports": args.ports,
        "supportPublicIp": True,
        "templateId": args.template_id,
    }
    if args.regions:
        payload["dataCenterIds"] = args.regions
        payload["dataCenterPriority"] = "availability"
    return payload


def is_retryable_create_failure(status: int, payload: Any) -> bool:
    message = extract_error_message(payload).lower()
    if status in (429, 500, 502, 503, 504):
        return True
    return "insufficient resources" in message or "capacity" in message


def print_pod_summary(pod: dict[str, Any]) -> None:
    pod_id = pod.get("id")
    name = pod.get("name")
    status = pod.get("desiredStatus")
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    ssh_port = port_mappings.get("22") or port_mappings.get(22)
    print(json.dumps(pod, indent=2, sort_keys=True))
    if public_ip and ssh_port:
        print()
        print(f"pod_id: {pod_id}")
        print(f"name: {name}")
        print(f"status: {status}")
        print(f"ssh: ssh root@{public_ip} -p {ssh_port}")


def wait_for_ssh(token: str, pod_id: str, *, timeout_seconds: int, poll_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        pod = get_pod(token, pod_id)
        if pod is None:
            raise RunpodApiError(f"pod {pod_id} disappeared while waiting for SSH")
        desired = pod.get("desiredStatus")
        if desired == "TERMINATED":
            raise RunpodApiError(f"pod {pod_id} terminated before SSH became ready")
        public_ip = pod.get("publicIp")
        port_mappings = pod.get("portMappings") or {}
        ssh_port = port_mappings.get("22") or port_mappings.get(22)
        if public_ip and ssh_port:
            return pod
        print(
            f"[wait] pod={pod_id} desiredStatus={desired} publicIp={public_ip!r} "
            f"ssh_port={ssh_port!r}; sleeping {poll_seconds}s",
            flush=True,
        )
        time.sleep(poll_seconds)
    raise RunpodApiError(f"timed out waiting for SSH on pod {pod_id}")


def acquire(args: argparse.Namespace) -> int:
    payload = build_create_payload(args)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    token = require_api_key()

    existing = list_existing_pods(token, args.name)
    live_existing = [pod for pod in existing if pod.get("desiredStatus") != "TERMINATED"]
    if live_existing:
        pod = live_existing[0]
        print(f"[reuse] found existing pod named {args.name}: {pod.get('id')}", flush=True)
        pod = wait_for_ssh(
            token,
            str(pod["id"]),
            timeout_seconds=args.ssh_timeout_minutes * 60,
            poll_seconds=args.ssh_poll_seconds,
        )
        print_pod_summary(pod)
        return 0

    deadline = time.time() + args.timeout_minutes * 60
    sleep_seconds = args.initial_sleep_seconds
    attempt = 1
    while time.time() < deadline:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        region_desc = args.regions if args.regions else "ANY_REGION"
        print(
            f"[attempt {attempt}] {now} creating {args.gpu_count}x {args.gpu_type} in {region_desc}",
            flush=True,
        )
        resp = request_json(token, "POST", "/pods", payload=payload)
        if resp.status in (200, 201):
            if not isinstance(resp.payload, dict) or "id" not in resp.payload:
                raise RunpodApiError(f"create pod succeeded but returned unexpected payload: {resp.payload!r}")
            pod_id = str(resp.payload["id"])
            print(f"[success] created pod {pod_id}", flush=True)
            pod = wait_for_ssh(
                token,
                pod_id,
                timeout_seconds=args.ssh_timeout_minutes * 60,
                poll_seconds=args.ssh_poll_seconds,
            )
            print_pod_summary(pod)
            return 0

        message = extract_error_message(resp.payload)
        if resp.status == 400 and "dataCenterIds" in payload:
            allowed_regions = extract_allowed_datacenter_ids(resp.payload)
            if allowed_regions:
                current_regions = [str(region) for region in payload["dataCenterIds"]]
                valid_regions = [region for region in current_regions if region in allowed_regions]
                invalid_regions = [region for region in current_regions if region not in allowed_regions]
                if invalid_regions and valid_regions:
                    print(
                        f"[sanitize] dropping invalid RunPod REST datacenter ids {invalid_regions}; "
                        f"continuing with {valid_regions}",
                        flush=True,
                    )
                    payload["dataCenterIds"] = valid_regions
                    args.regions = valid_regions
                    attempt += 1
                    continue
        if not is_retryable_create_failure(resp.status, resp.payload):
            raise RunpodApiError(f"create pod failed with status {resp.status}: {message}")

        remaining = int(deadline - time.time())
        if remaining <= 0:
            break
        next_sleep = min(sleep_seconds, remaining)
        print(
            f"[retryable] status={resp.status} message={message!r}; "
            f"sleeping {next_sleep}s (remaining budget {remaining}s)",
            flush=True,
        )
        time.sleep(next_sleep)
        sleep_seconds = min(args.max_sleep_seconds, sleep_seconds + args.sleep_increment_seconds)
        attempt += 1

    print("timed out without acquiring an H100 pod", file=sys.stderr)
    return 1


def status_cmd(args: argparse.Namespace) -> int:
    token = require_api_key()
    pod = get_pod(token, args.pod_id)
    if pod is None:
        print(f"pod {args.pod_id} not found")
        return 1
    print_pod_summary(pod)
    return 0


def delete_cmd(args: argparse.Namespace) -> int:
    token = require_api_key()
    delete_pod(token, args.pod_id)
    print(f"deleted pod {args.pod_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    acquire_p = sub.add_parser("acquire", help="retry H100 allocation until capacity appears")
    acquire_p.add_argument("--name", default="parameter-golf-h100x8", help="RunPod pod name")
    acquire_p.add_argument(
        "--gpu-type",
        default=DEFAULT_GPU_TYPE,
        help="RunPod GPU type id or label; default is the H100 SXM SKU id",
    )
    acquire_p.add_argument("--gpu-count", type=int, default=8, help="GPU count on the pod")
    acquire_p.add_argument(
        "--regions",
        nargs="+",
        default=DEFAULT_REGIONS,
        help="Optional RunPod data center ids to restrict placement; omit for Any region",
    )
    acquire_p.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID, help="RunPod template id")
    acquire_p.add_argument("--container-disk-gb", type=int, default=100, help="container disk size")
    acquire_p.add_argument("--volume-gb", type=int, default=20, help="persistent volume size")
    acquire_p.add_argument("--volume-mount-path", default="/workspace", help="volume mount path")
    acquire_p.add_argument(
        "--ports",
        nargs="+",
        default=["22/tcp"],
        help="ports to expose on the pod",
    )
    acquire_p.add_argument("--timeout-minutes", type=int, default=180, help="overall retry timeout")
    acquire_p.add_argument("--initial-sleep-seconds", type=int, default=60, help="initial wait between retries")
    acquire_p.add_argument("--sleep-increment-seconds", type=int, default=30, help="backoff increment")
    acquire_p.add_argument("--max-sleep-seconds", type=int, default=300, help="max wait between retries")
    acquire_p.add_argument("--ssh-timeout-minutes", type=int, default=20, help="wait budget after allocation")
    acquire_p.add_argument("--ssh-poll-seconds", type=int, default=15, help="SSH readiness poll interval")
    acquire_p.add_argument("--dry-run", action="store_true", help="print the create request body and exit")
    acquire_p.set_defaults(func=acquire)

    status_p = sub.add_parser("status", help="show pod status")
    status_p.add_argument("pod_id")
    status_p.set_defaults(func=status_cmd)

    delete_p = sub.add_parser("delete", help="delete a pod by id")
    delete_p.add_argument("pod_id")
    delete_p.set_defaults(func=delete_cmd)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except RunpodApiError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

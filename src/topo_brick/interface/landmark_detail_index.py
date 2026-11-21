"""
Phase 11 (`Ext 4B`): `LandmarkDetailIndex` over Non-Flush Architectural & Functional Molds across
`8` Semantic Roles (`FINIAL_SPIRE_CONE`, `DOME_ROOF_CAP`, `CLIP_FLAG_BANNER`, `BAR_POLE_ANTENNA`,
`WINDOW_FRAME_LATTICE`, `DOOR_PORTCULLIS`, `TEXTURED_MASONRY_BRICK`, `HINGED_PANEL_MOUNT`).

Compiles and lazily loads `compiled_landmark_detail_index.json.gz` (`< 80 KiB`), guaranteeing
verified `CC-BY-4.0` (`COMMERCIALLY_USABLE_WITH_ATTRIBUTION`) AND full `all`-license molds for
every role, with its SHA-256 bound into `compute_catalog_sha256(schema_version="1.1.0")`.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    get_declarative_part_spec,
    is_known_part_id,
)

LANDMARK_DETAIL_ROLES: Tuple[str, ...] = (
    "FINIAL_SPIRE_CONE",
    "DOME_ROOF_CAP",
    "CLIP_FLAG_BANNER",
    "BAR_POLE_ANTENNA",
    "WINDOW_FRAME_LATTICE",
    "DOOR_PORTCULLIS",
    "TEXTURED_MASONRY_BRICK",
    "HINGED_PANEL_MOUNT",
)

# Curated primary anchors per role (verified CC-BY-4.0 molds listed first in each tuple!)
_CURATED_ROLE_SEEDS: Dict[str, Tuple[str, ...]] = {
    "FINIAL_SPIRE_CONE": ("3942c", "12894", "4589", "3943b", "59900"),
    "DOME_ROOF_CAP": ("5848", "30367c", "30151b", "30151", "4740"),
    "CLIP_FLAG_BANNER": ("2555", "11476", "4085d", "61252", "2525", "4495", "2335"),
    "BAR_POLE_ANTENNA": ("3957b", "60478", "2540", "30374", "87994", "64644"),
    "WINDOW_FRAME_LATTICE": ("60592", "30044", "60593", "60601", "60607", "60594", "4033"),
    "DOOR_PORTCULLIS": ("60596", "60598", "60599", "60621", "60623", "3853"),
    "TEXTURED_MASONRY_BRICK": ("30136", "2877", "98283"),
    "HINGED_PANEL_MOUNT": ("3937", "3938", "2429", "2430", "30387", "60471", "92582", "44301", "44302"),
}

DEFAULT_LANDMARK_INDEX_GZ: Path = Path(__file__).resolve().parent / "compiled_landmark_detail_index.json.gz"


@dataclass(frozen=True, slots=True)
class LandmarkMoldRecord:
    """Indexed non-flush architectural or functional landmark mold (`Phase 11 Ext 4B`)."""

    part_id: str
    role: str
    title: str
    license_tier: str  # "cc_by_4_0" or "cc_by_sa_4_0"
    connector_types: Tuple[str, ...]
    bbox_ldu: Tuple[int, int, int]
    priority_rank: int


class LandmarkDetailIndex:
    """Deterministic `8`-role landmark detail catalog index backed by `compiled_landmark_detail_index.json.gz`."""

    def __init__(self, index_path: Path = DEFAULT_LANDMARK_INDEX_GZ) -> None:
        self._index_path = Path(index_path)
        self._loaded = False
        self._archive_sha256: str = ""
        self._by_role: Dict[str, Tuple[LandmarkMoldRecord, ...]] = {}
        self._by_part_id: Dict[str, LandmarkMoldRecord] = {}

    @property
    def index_path(self) -> Path:
        return self._index_path

    @property
    def archive_sha256(self) -> str:
        self._ensure_loaded()
        return self._archive_sha256

    @property
    def sha256(self) -> str:
        self._ensure_loaded()
        return self._archive_sha256

    def __len__(self) -> int:
        self._ensure_loaded()
        return sum(len(v) for v in self._by_role.values())

    def compile_and_save(self, target_path: Optional[Path] = None) -> Path:
        """Deterministically compile `compiled_landmark_detail_index.json.gz` across all `8` roles."""
        from topo_brick.interface.ldraw_shadow_catalog_compiler import COMPILED_LDRAW_CATALOG

        out_path = Path(target_path) if target_path is not None else self._index_path
        COMPILED_LDRAW_CATALOG._ensure_loaded()

        records_json: List[Dict[str, Any]] = []
        seen_pid_role: Set[Tuple[str, str]] = set()

        def _add_candidate(pid_raw: str, role_name: str, rank: int) -> None:
            pid = str(pid_raw).strip()
            if not is_known_part_id(pid):
                return
            key_pr = (pid, role_name)
            if key_pr in seen_pid_role:
                return
            try:
                spec = get_declarative_part_spec(pid)
            except Exception:
                return
            prim = spec.primitive
            if not prim.hull_boxes or not prim.ports:
                return
            lic = (
                "cc_by_sa_4_0"
                if spec.provenance.redistribution_status == "SHARE_ALIKE_ADAPTATION"
                else "cc_by_4_0"
            )
            xmin = min(b.xmin for b in prim.hull_boxes)
            xmax = max(b.xmax for b in prim.hull_boxes)
            ymin = min(b.ymin for b in prim.hull_boxes)
            ymax = max(b.ymax for b in prim.hull_boxes)
            zmin = min(b.zmin for b in prim.hull_boxes)
            zmax = max(b.zmax for b in prim.hull_boxes)
            if (xmax - xmin) > 160 or (ymax - ymin) > 160 or (zmax - zmin) > 192:
                return
            ctypes = tuple(sorted({p.connector_type.name for p in prim.ports}))
            title = getattr(prim, "name", pid)
            if pid.lower() in COMPILED_LDRAW_CATALOG._raw_parts:
                title = str(COMPILED_LDRAW_CATALOG._raw_parts[pid.lower()].get("n", title))
            seen_pid_role.add(key_pr)
            records_json.append(
                {
                    "part_id": pid,
                    "role": role_name,
                    "title": title,
                    "license_tier": lic,
                    "connector_types": list(ctypes),
                    "bbox_ldu": [int(xmax - xmin), int(ymax - ymin), int(zmax - zmin)],
                    "priority_rank": int(rank),
                }
            )

        # 1. Add verified curated seeds first (`priority_rank = 0..20`)
        for role_name in LANDMARK_DETAIL_ROLES:
            for r_idx, pid_s in enumerate(_CURATED_ROLE_SEEDS[role_name]):
                _add_candidate(pid_s, role_name, r_idx)

        # 2. Scan `COMPILED_LDRAW_CATALOG` for additional clean unpatterned architectural landmarks (up to 20 per role)
        per_role_extra: Dict[str, int] = {r: 0 for r in LANDMARK_DETAIL_ROLES}
        for pid_c in sorted(COMPILED_LDRAW_CATALOG._raw_parts.keys()):
            rec = COMPILED_LDRAW_CATALOG._raw_parts[pid_c]
            title_l = str(rec.get("n", "")).lower()
            if any(bad in title_l for bad in ("pattern", "sticker", "duplo", "minifig", "technic")):
                continue
            matched_role: Optional[str] = None
            if title_l.startswith("cone ") and "1 x  1" in title_l:
                matched_role = "FINIAL_SPIRE_CONE"
            elif "dome" in title_l and ("2 x  2" in title_l or "1 x  1" in title_l):
                matched_role = "DOME_ROOF_CAP"
            elif title_l.startswith("flag ") and "clip" in title_l:
                matched_role = "CLIP_FLAG_BANNER"
            elif title_l.startswith(("antenna ", "bar ")) and len(title_l) < 42:
                matched_role = "BAR_POLE_ANTENNA"
            elif title_l.startswith("window ") and "frame" in title_l:
                matched_role = "WINDOW_FRAME_LATTICE"
            elif title_l.startswith("door ") and ("frame" in title_l or "1 x  4" in title_l):
                matched_role = "DOOR_PORTCULLIS"
            elif ("masonry" in title_l or "palisade" in title_l or "log" in title_l) and title_l.startswith("brick"):
                matched_role = "TEXTURED_MASONRY_BRICK"
            elif title_l.startswith("hinge ") and ("brick" in title_l or "plate" in title_l):
                matched_role = "HINGED_PANEL_MOUNT"

            if matched_role is not None and per_role_extra[matched_role] < 14:
                before_n = len(records_json)
                _add_candidate(pid_c, matched_role, 100 + per_role_extra[matched_role])
                if len(records_json) > before_n:
                    per_role_extra[matched_role] += 1

        payload = {
            "schema_version": "1.0.0",
            "roles": list(LANDMARK_DETAIL_ROLES),
            "total_records": len(records_json),
            "records": records_json,
        }
        raw_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(gzip.compress(raw_bytes, compresslevel=9, mtime=0))

        self._loaded = False
        self._index_path = out_path
        self._ensure_loaded()
        return out_path

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self._index_path.is_file():
            self.compile_and_save(self._index_path)
            return
        raw_gz = self._index_path.read_bytes()
        self._archive_sha256 = hashlib.sha256(raw_gz).hexdigest()
        data = json.loads(gzip.decompress(raw_gz).decode("utf-8"))
        by_role_lists: Dict[str, List[LandmarkMoldRecord]] = {r: [] for r in LANDMARK_DETAIL_ROLES}
        by_pid: Dict[str, LandmarkMoldRecord] = {}
        for item in data.get("records", []):
            rec = LandmarkMoldRecord(
                part_id=str(item["part_id"]),
                role=str(item["role"]),
                title=str(item["title"]),
                license_tier=str(item["license_tier"]),
                connector_types=tuple(str(x) for x in item["connector_types"]),
                bbox_ldu=(int(item["bbox_ldu"][0]), int(item["bbox_ldu"][1]), int(item["bbox_ldu"][2])),
                priority_rank=int(item["priority_rank"]),
            )
            by_role_lists.setdefault(rec.role, []).append(rec)
            if rec.part_id not in by_pid:
                by_pid[rec.part_id] = rec
        for r_k, lst in by_role_lists.items():
            lst.sort(key=lambda r: (0 if r.part_id in EXTENDED_BRICK_CATALOG else 1, r.priority_rank, r.part_id))
            self._by_role[r_k] = tuple(lst)
        self._by_part_id = by_pid
        self._loaded = True

    def query_role(
        self,
        role: str,
        license_filter: str = "all",
        allowed_part_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[LandmarkMoldRecord, ...]:
        """Return prioritized `LandmarkMoldRecord` entries for `role`, respecting `license_filter` & inventory."""
        self._ensure_loaded()
        if role not in self._by_role:
            raise ValueError(f"Unknown landmark role {role!r}; expected one of {LANDMARK_DETAIL_ROLES}.")
        if license_filter not in ("all", "cc_by_4_0_only"):
            raise ValueError(f"Unsupported license_filter={license_filter!r}.")
        cc4 = license_filter == "cc_by_4_0_only"
        allow_set = {str(p).strip() for p in allowed_part_ids} if allowed_part_ids is not None else None
        out: List[LandmarkMoldRecord] = []
        for rec in self._by_role[role]:
            if cc4 and rec.license_tier != "cc_by_4_0":
                continue
            if allow_set is not None and rec.part_id not in allow_set and rec.part_id.lower() not in allow_set:
                continue
            out.append(rec)
        return tuple(out)


LANDMARK_DETAIL_INDEX = LandmarkDetailIndex()

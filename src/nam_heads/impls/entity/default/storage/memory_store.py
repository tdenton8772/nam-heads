# src/encoders/heads/entity/storage/memory_store.py

from typing import Optional, Dict, Any, List, Tuple
from .base import EntityStore


class MemoryEntityStore(EntityStore):
    def __init__(self):
        self.surface_to_entity: Dict[str, str] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        # Candidate key index: key -> [(entity_id, surface_form), ...]
        self.candidate_key_index: Dict[str, List[Tuple[str, str]]] = {}
        # Redirect forward pointers: old_id -> new_id
        self._redirects: Dict[str, str] = {}
        # Redirect reverse index: canonical_id -> [old_ids]
        self._reverse_redirects: Dict[str, List[str]] = {}
        # Raw documents (get_raw_batch / put_raw_batch).
        self.raw: Dict[str, bytes] = {}

    def resolve_surface_form(self, surface: str, entity_type: Optional[str] = None) -> Optional[str]:
        if entity_type:
            # Type-scoped lookup only -- no unscoped fallback to prevent cross-type pollution
            return self.surface_to_entity.get(f"{entity_type}::{surface}")
        # Unscoped lookup (primary forms only)
        return self.surface_to_entity.get(surface)

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.entities.get(entity_id)

    def create_entity(
        self,
        entity_id: str,
        surface_forms: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        if entity_id not in self.entities:
            self.entities[entity_id] = {
                "entity_id": entity_id,
                "surface_forms": [],
                "metadata": metadata,
            }

        entity_type = metadata.get("entity_type")
        primary_form = surface_forms[0] if surface_forms else None
        for sf in surface_forms:
            self._register_surface_form(entity_id, sf, entity_type, is_primary=(sf == primary_form))

    def _register_surface_form(self, entity_id: str, surface_form: str,
                                entity_type: Optional[str], is_primary: bool) -> None:
        """Register a surface form with type-scoped and optionally unscoped keys."""
        entity = self.entities.get(entity_id)
        if entity:
            sfs = entity["surface_forms"]
            if surface_form not in sfs:
                sfs.append(surface_form)

        # Type-scoped key (first-writer-wins)
        if entity_type:
            scoped_key = f"{entity_type}::{surface_form}"
            if scoped_key not in self.surface_to_entity:
                self.surface_to_entity[scoped_key] = entity_id

        # Unscoped key only for primary form (first-writer-wins)
        if is_primary:
            if surface_form not in self.surface_to_entity:
                self.surface_to_entity[surface_form] = entity_id

    def add_surface_form(
        self,
        entity_id: str,
        surface_form: str,
        max_forms: int,
    ) -> None:
        entity = self.entities.get(entity_id)
        if not entity:
            return

        entity_type = entity.get("metadata", {}).get("entity_type")
        sfs = entity["surface_forms"]
        if surface_form not in sfs:
            sfs.append(surface_form)
            if len(sfs) > max_forms:
                del sfs[0]

        # Register type-scoped key
        if entity_type:
            scoped_key = f"{entity_type}::{surface_form}"
            if scoped_key not in self.surface_to_entity:
                self.surface_to_entity[scoped_key] = entity_id

    def resolve_surface_forms_batch(self, queries):
        result = {}
        for surface, entity_type in queries:
            result[(surface, entity_type)] = self.resolve_surface_form(surface, entity_type)
        return result

    def create_entities_batch(self, entities):
        for entity_id, surface_forms, metadata in entities:
            self.create_entity(entity_id, surface_forms, metadata)

    def get_raw_batch(self, keys):
        return {k: self.raw[k] for k in keys if k in self.raw}

    def put_raw_batch(self, items):
        for k, v in items:
            self.raw[k] = v

    def lookup_candidate_keys_by_key(self, keys):
        result = {}
        for key in keys:
            result[key] = list(self.candidate_key_index.get(key, []))
        return result

    def store_candidate_keys(
        self,
        entity_id: str,
        surface_form: str,
        keys: List[str],
    ) -> None:
        pair = (entity_id, surface_form)
        for key in keys:
            bucket = self.candidate_key_index.setdefault(key, [])
            if pair not in bucket:
                bucket.append(pair)

    def store_candidate_keys_bulk(
        self,
        items: List[Tuple[str, str, List[str]]],
    ) -> None:
        """Batch store candidate keys for multiple (entity_id, surface_form, keys) tuples."""
        for entity_id, surface_form, keys in items:
            self.store_candidate_keys(entity_id, surface_form, keys)

    def lookup_candidate_keys(
        self,
        keys: List[str],
    ) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        for key in keys:
            results.extend(self.candidate_key_index.get(key, []))
        return results

    # --- Redirect support ---

    def write_redirect(self, old_entity_id: str, new_entity_id: str) -> None:
        if old_entity_id == new_entity_id:
            return
        existing = self._redirects.get(old_entity_id)
        if existing == new_entity_id:
            return  # idempotent
        self._redirects[old_entity_id] = new_entity_id
        rev_list = self._reverse_redirects.setdefault(new_entity_id, [])
        if old_entity_id not in rev_list:
            rev_list.append(old_entity_id)

    def resolve_redirect(self, entity_id: str) -> str:
        visited = set()
        current = entity_id
        for _ in range(10):
            if current in visited:
                break  # cycle
            visited.add(current)
            next_id = self._redirects.get(current)
            if next_id is None:
                break
            current = next_id
        return current

    def reverse_redirects(self, canonical_id: str) -> List[str]:
        result = []
        queue = list(self._reverse_redirects.get(canonical_id, []))
        visited = {canonical_id}
        while queue:
            old_id = queue.pop(0)
            if old_id in visited:
                continue
            visited.add(old_id)
            result.append(old_id)
            # Transitive: if something redirects to old_id, include it too
            queue.extend(self._reverse_redirects.get(old_id, []))
        return result

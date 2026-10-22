from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple


class EntityStore(ABC):
    """
    Canonical interface for entity storage.

    ALL backends must implement this interface exactly.
    """

    @abstractmethod
    def resolve_surface_form(self, surface: str, entity_type: Optional[str] = None) -> Optional[str]:
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def create_entity(
        self,
        entity_id: str,
        surface_forms: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def add_surface_form(
        self,
        entity_id: str,
        surface_form: str,
        max_forms: int,
    ) -> None:
        pass

    # --- Fuzzy resolution support ---

    @abstractmethod
    def store_candidate_keys(
        self,
        entity_id: str,
        surface_form: str,
        keys: List[str],
    ) -> None:
        """Store reverse-index candidate keys for fuzzy lookup."""
        pass

    @abstractmethod
    def lookup_candidate_keys(
        self,
        keys: List[str],
    ) -> List[Tuple[str, str]]:
        """Lookup (entity_id, surface_form) pairs matching any of the given keys."""
        pass

    # --- Redirect support (entity ID migration) ---

    @abstractmethod
    def write_redirect(self, old_entity_id: str, new_entity_id: str) -> None:
        """Write a forward redirect pointer: old_entity_id → new_entity_id.

        Also updates the reverse index so reverse_redirects(new_entity_id)
        includes old_entity_id.  Idempotent — skip if redirect already exists.
        """
        pass

    @abstractmethod
    def resolve_redirect(self, entity_id: str) -> str:
        """Follow redirect chain to canonical entity ID.

        Returns entity_id unchanged if no redirect exists.
        Max 10 hops, cycle detection via visited set.
        """
        pass

    @abstractmethod
    def reverse_redirects(self, canonical_id: str) -> List[str]:
        """Return all old entity IDs that redirect (transitively) to canonical_id.

        BFS traversal of redir_rev:: reverse index.
        """
        pass

    def resolve_surface_forms_batch(
        self,
        queries: "List[tuple]",
    ) -> "Dict[tuple, Optional[str]]":
        """Batch resolve multiple (surface, entity_type) pairs.

        Default: serial fallback.  CouchbaseKV overrides with one batch_get.
        Returns dict mapping (surface, entity_type) -> entity_id or None.
        """
        result = {}
        for surface, entity_type in queries:
            result[(surface, entity_type)] = self.resolve_surface_form(surface, entity_type)
        return result

    def create_entities_batch(
        self,
        entities: "List[tuple]",
    ) -> None:
        """Batch create multiple entities.

        Default: serial fallback.  CouchbaseKV overrides with pipelined writes.
        Each tuple: (entity_id, surface_forms, metadata).
        """
        for entity_id, surface_forms, metadata in entities:
            self.create_entity(entity_id, surface_forms, metadata)

    def lookup_candidate_keys_by_key(
        self,
        keys: "List[str]",
    ) -> "Dict[str, List[tuple]]":
        """Lookup candidate keys, returning per-key results.

        Default: serial fallback.  CouchbaseKV overrides with one batch_get.
        Returns dict mapping query_key -> [(entity_id, surface_form), ...].
        """
        result: dict = {}
        for key in keys:
            pairs = self.lookup_candidate_keys([key])
            result[key] = pairs
        return result

    def get_raw_batch(self, keys: "List[str]") -> "Dict[str, bytes]":
        """Raw documents by key (the entity head's own keys outside the
        surface-form index, e.g. cpt:: concept state). Absent keys are left
        out. Default: none."""
        return {}

    def put_raw_batch(self, items: "List[Tuple[str, bytes]]") -> None:
        """Write raw documents (see ``get_raw_batch``). Default: dropped."""
        return None

    def resolve_fuzzy_batch(
        self,
        queries: "List[Tuple[str, str, Any]]",
        max_candidates: int = 50,
    ) -> "Dict[int, Optional[str]]":
        """Batch fuzzy resolution.

        Parameters
        ----------
        queries : list of (canonical, entity_type, comparator)
        max_candidates : max candidates per query

        Returns dict mapping query index -> entity_id or None.
        Default: serial fallback using resolve_fuzzy().
        """
        result: Dict[int, Optional[str]] = {}
        for i, (surface, entity_type, comparator) in enumerate(queries):
            result[i] = self.resolve_fuzzy(surface, entity_type, comparator, max_candidates)
        return result

    def resolve_fuzzy(
        self,
        surface: str,
        entity_type: str,
        comparator: Any,
        max_candidates: int = 50,
    ) -> Optional[str]:
        """
        Fuzzy entity resolution: generate query keys, fetch candidates, score, return best.

        Returns entity_id if a match above comparator.threshold() is found, else None.
        """
        query_keys = comparator.generate_keys(surface)
        if not query_keys:
            return None

        candidates = self.lookup_candidate_keys(query_keys)
        if not candidates:
            return None

        # Deduplicate and limit
        seen = set()
        unique = []
        for entity_id, candidate_surface in candidates:
            if entity_id not in seen:
                seen.add(entity_id)
                unique.append((entity_id, candidate_surface))
            if len(unique) >= max_candidates:
                break

        best_id = None
        best_score = 0.0
        threshold = comparator.threshold()

        for entity_id, candidate_surface in unique:
            score = comparator.match(surface, candidate_surface)
            # Equal scores resolve to the smallest id, not to write order.
            if score >= threshold and (
                    score > best_score
                    or (score == best_score and best_id is not None and entity_id < best_id)):
                best_score = score
                best_id = entity_id

        return best_id
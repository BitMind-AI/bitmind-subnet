import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import bittensor as bt
import numpy as np

from gas.cache.content_db import ContentDB
from gas.cache.media_storage import MediaStorage
from gas.cache.types import Media, MediaEntry, PromptEntry
from gas.cache.util import extract_media_info, get_format_from_content
from gas.types import MediaType, Modality


class ContentManager:
	"""
	Unified interface for managing content storage and retrieval.
    Encapsulates both filesystem and database operations,
	Provides high-level methods like write_media() and write_prompt().
	"""

	def __init__(
		self,
		base_dir: Optional[Path] = None,
		max_per_source: int = 500,
		enable_source_limits: bool = True,
		prune_strategy: str = "oldest",
		remove_on_sample: bool = True,
		min_source_threshold: float = 0.8
	):
		"""
		Initialize the content manager.

		Args:
			base_dir: Base directory for cache storage (defaults to ~/.cache/sn34)
			max_per_source: Maximum items per source (dataset/scraper/model)
			enable_source_limits: Whether to enable source limits
			prune_strategy: Strategy for pruning ('oldest', 'least_used', 'random')
			remove_on_sample: Whether to remove items when sampled
			min_source_threshold: Minimum items before triggering download
		"""
		if base_dir is None:
			base_dir = Path("~/.cache/sn34").expanduser()

		self.base_dir = Path(base_dir)
		self.content_db = ContentDB(self.base_dir / "prompts.db")
		self.media_storage = MediaStorage(self.base_dir)

		self.max_per_source = max_per_source
		self.enable_source_limits = enable_source_limits
		self.prune_strategy = prune_strategy
		
		self.remove_on_sample = remove_on_sample
		min_source_threshold = 0.8 if min_source_threshold is None else min_source_threshold
		self.min_source_threshold = int(max_per_source * float(min_source_threshold))

	def write_prompt(self, content: str, content_type: str = "prompt", source_media_id: Optional[str] = None) -> str:
		try:
			prompt_id = self.content_db.add_prompt_entry(
				content=content,
				content_type=content_type,
				source_media_id=source_media_id
			)
			bt.logging.debug(f"Added {content_type} to database with ID: {prompt_id}")
			return prompt_id
		except Exception as e:
			bt.logging.error(f"Error writing {content_type} to database: {e}")
			raise

	def write_generated_media(
		self,
		modality: Modality,
		media_type: MediaType,
		model_name: str,
		prompt_id: str,
		media_content: Any,
		mask_content: Optional[np.ndarray] = None,
		generation_args: Optional[Dict[str, Any]] = None,
	) -> Optional[str]:
		"""
		Write generated media (from models) to storage.

		Args:
			modality: Modality.IMAGE or Modality.VIDEO
			media_type: MediaType.REAL, MediaType.SYNTHETIC, or MediaType.SEMISYNTHETIC
			model_name: Name of the model that generated this media
			prompt_id: ID of the associated prompt
			media_content: The media content (PIL Image, video frames, etc.)
			mask_content: Optional mask for images
			generation_args: args used in generating this media
		Returns:
			Path to the saved media file, or None if failed
		"""
		try:
			format = get_format_from_content(media_content, modality)

			media_data = Media(
				modality=modality,
				media_type=media_type,
				prompt_id=prompt_id,
				media_content=media_content,
				format=format,
				model_name=model_name,
				mask_content=mask_content,
				generation_args=generation_args
			)

			save_path, mask_path = self.media_storage.write_media(media_data)

			if save_path is None:
				bt.logging.error("Failed to write media to filesystem")
				return None

			resolution, file_size = extract_media_info(save_path, media_data.modality)

			# Add entry to database
			media_id = self.content_db.add_media_entry(
				prompt_id=media_data.prompt_id,
				file_path=save_path,
				modality=media_data.modality,
				media_type=media_data.media_type,
				source_type="generated",
				generation_args=generation_args,
				model_name=media_data.model_name,
				mask_path=mask_path,
				timestamp=int(time.time()),
				resolution=resolution,
				file_size=file_size,
				format=media_data.format,
			)

			bt.logging.info(f"Saved media to {save_path} with database ID: {media_id}")
			return save_path

		except Exception as e:
			bt.logging.error(f"Error writing media: {e}")
			return None

	def write_scraped_media(
		self,
		modality: Modality,
		media_type: MediaType,
		prompt_id: str,
		media_content: Any,
		download_url: str,
		scraper_name: str,
		mask_content: Optional[np.ndarray] = None,
		resolution: Optional[tuple[int, int]] = None,
	) -> Optional[str]:
		"""
		Write scraped media (from web sources) to storage.

		Args:
			modality: Modality.IMAGE or Modality.VIDEO
			media_type: MediaType.REAL or MediaType.SEMISYNTHETIC
			prompt_id: ID of the associated search query
			media_content: The media content (PIL Image, video frames, etc.)
			download_url: URL where the media was scraped from
			scraper_name: Name of the scraper (e.g., 'google', 'bing')
			mask_content: Optional mask for images
			resolution: Optional (width, height) tuple

		Returns:
			Path to the saved media file, or None if failed
		"""
		media_data = Media(
			modality=modality,
			media_type=media_type,
			prompt_id=prompt_id,
			media_content=media_content,
			format=get_format_from_content(media_content, modality),
			model_name=None,  # Not applicable for scraped media
			mask_content=mask_content,
		)

		save_path, mask_path = self.media_storage.write_media(media_data)

		if save_path is None:
			bt.logging.error("Failed to write scraped media to filesystem")
			return None

		# Use provided resolution if available, otherwise extract from file
		if resolution is None:
			resolution, file_size = extract_media_info(save_path, media_data.modality)
		else:
			file_size = extract_media_info(save_path, media_data.modality)[1]

		# Add entry to database with source_type='scraper'
		media_id = self.content_db.add_media_entry(
			prompt_id=media_data.prompt_id,
			file_path=save_path,
			modality=media_data.modality,
			media_type=media_data.media_type,
			source_type="scraper",
			download_url=download_url,
			scraper_name=scraper_name,
			mask_path=mask_path,
			timestamp=int(time.time()),
			resolution=resolution,
			file_size=file_size,
			format=media_data.format,
		)

		bt.logging.info(
			f"Saved scraped media to {save_path} with database ID: {media_id}"
		)
		return save_path

	def write_dataset_media(
		self,
		modality: Modality,
		media_type: MediaType,
		media_content: Any,
		dataset_name: str,
		dataset_source_file: str,
		dataset_index: str,
		mask_content: Optional[np.ndarray] = None,
		resolution: Optional[tuple[int, int]] = None,
	) -> Optional[str]:
		"""
		Write dataset media (from HuggingFace datasets) to storage.

		Args:
			modality: Modality.IMAGE or Modality.VIDEO
			media_type: MediaType.REAL (datasets are real media)
			media_content: The media content (bytes, PIL Image, etc.)
			dataset_name: Name/path of the dataset (e.g., 'laion/laion2B-en')
			dataset_source_file: Source file within dataset (e.g., 'data_001.parquet')
			dataset_index: Index within the dataset file
			mask_content: Optional mask for images
			resolution: Optional (width, height) tuple

		Returns:
			Path to the saved media file, or None if failed
		"""
		media_data = Media(
			modality=modality,
			media_type=media_type,
			prompt_id=None,  # Dataset media is not tied to prompts
			media_content=media_content,
			format=get_format_from_content(media_content, modality),
			model_name=None,  # Not applicable for dataset media
			mask_content=mask_content,
		)

		save_path, mask_path = self.media_storage.write_media(media_data)

		if save_path is None:
			bt.logging.error("Failed to write dataset media to filesystem")
			return None

		# Use provided resolution if available, otherwise extract from file
		if resolution is None:
			resolution, file_size = extract_media_info(save_path, media_data.modality)
		else:
			file_size = extract_media_info(save_path, media_data.modality)[1]

		# Add entry to database with source_type='dataset'
		media_id = self.content_db.add_media_entry(
			prompt_id=None,  # Dataset media is not tied to prompts
			file_path=save_path,
			modality=media_data.modality,
			media_type=media_data.media_type,
			source_type="dataset",
			dataset_name=dataset_name,
			dataset_source_file=dataset_source_file,
			dataset_index=dataset_index,
			mask_path=mask_path,
			timestamp=int(time.time()),
			resolution=resolution,
			file_size=file_size,
			format=media_data.format,
		)

		bt.logging.info(
			f"Saved dataset media to {save_path} with database ID: {media_id}"
		)
		return save_path

	def sample_prompts(
        self, k: 
        int = 1, 
        remove: bool = True, 
        strategy: str = "random") -> List[PromptEntry]:
		return self.content_db.sample_prompt_entries(k=k, remove=remove, strategy=strategy, content_type="prompt")

	def sample_search_queries(
        self, 
        k: int = 1, 
        remove: bool = False, 
        strategy: str = "random") -> List[PromptEntry]:
		return self.content_db.sample_prompt_entries(k=k, remove=remove, strategy=strategy, content_type="search_query")

	def sample_media(
		self,
		k: int = 1,
		modality: Modality = Modality.IMAGE,
		media_type: MediaType = MediaType.SYNTHETIC,
		remove: bool = False,
		strategy: str = "random"
	) -> List[MediaEntry]:

		return self.content_db.sample_media_entries(
			k=k,
			modality=modality,
			media_type=media_type,
			strategy=strategy,
			remove=remove
		)

	def sample_media_with_content(
		self,
		modality: Modality,
		media_type: MediaType,
		count: int = 1,
		remove_from_cache: bool = None,
		**kwargs,
	) -> Optional[Dict[str, Any]]:
        
		should_remove = remove_from_cache if remove_from_cache is not None else self.remove_on_sample
		media_entries = self.content_db.sample_media_entries(
			k=count,
			modality=modality,
			media_type=media_type,
			strategy=kwargs.get("strategy", "random"),
			remove=should_remove,
		)
		if not media_entries:
			print(f"No media available in database for {modality}/{media_type}")
			return {'count': 0, 'items': []}

		# Split by source_type so we only remove dataset/scraper on sample
		generated_entries: List[MediaEntry] = [e for e in media_entries if getattr(e, 'source_type', None) == 'generated']
		non_generated_entries: List[MediaEntry] = [e for e in media_entries if getattr(e, 'source_type', None) != 'generated']

		items: List[Dict[str, Any]] = []
		# Retrieve non-generated content; allow removal if configured
		if non_generated_entries:
			non_gen_items = self.media_storage.retrieve_media(
				media_entries=non_generated_entries,
				modality=modality,
				remove_from_cache=should_remove,
				**kwargs,
			)['items']
			items.extend(non_gen_items)

		# Retrieve generated content; never remove on sample
		if generated_entries:
			gen_items = self.media_storage.retrieve_media(
				media_entries=generated_entries,
				modality=modality,
				remove_from_cache=False,
				**kwargs,
			)['items']
			items.extend(gen_items)

		# Attach ids and metadata; order doesn't matter, so align by concatenation
		combined_entries: List[MediaEntry] = non_generated_entries + generated_entries
		for media, db_entry in zip(items, combined_entries):
			media['id'] = db_entry.id
			media['metadata'] = db_entry.to_dict()

		return {'count': len(items), 'items': items}

	def sample_prompts_with_source_media(
		self,
		k: int = 1,
		remove: bool = True,
		strategy: str = "random",
		**kwargs
	) -> List[Dict[str, Any]]:
		"""
		Sample prompts and load their associated source media content (first item only).
		Returns a list of { 'prompt': PromptEntry, 'media': media_item }.
		"""
		prompt_entries = self.content_db.sample_prompt_entries(
			k=k, remove=remove, strategy=strategy, content_type="prompt"
		)
		results: List[Dict[str, Any]] = []
		for prompt in prompt_entries:
			if not prompt.source_media_id:
				continue
			media_entries = self.content_db.get_media_entries(media_id=prompt.source_media_id)
			if not media_entries:
				continue
			media_content = self.media_storage.retrieve_media(
				media_entries=media_entries,
				modality=Modality.IMAGE,
				**kwargs
			)
			if media_content.get("count", 0) > 0:
				results.append({
					"prompt": prompt,
					"media": media_content["items"][0]
				})
		return results

	def _check_and_prune_source(self, source_type: str, source_name: str, max_count: int) -> int:
		if not self.enable_source_limits:
			return 0
		current_count = self.content_db.get_source_count(source_type, source_name)
		if current_count >= max_count:
			pruned = self.content_db.prune_source_media(
				source_type, source_name, max_count, self.prune_strategy
			)
			if pruned > 0:
				bt.logging.info(f"Pruned {pruned} items from {source_type} '{source_name}' to enforce cap {max_count}")
			return pruned
		return 0

	def enforce_source_caps(self) -> Dict[str, int]:
		results: Dict[str, int] = {}
		if not self.enable_source_limits:
			return results
		try:
			counts = self.content_db.get_source_counts()
			for source_type, sources in counts.items():
				for source_name, count in sources.items():
					if count > self.max_per_source:
						pruned = self.content_db.prune_source_media(
							source_type, source_name, self.max_per_source, self.prune_strategy
						)
						if pruned > 0:
							key = f"{source_type}:{source_name}"
							results[key] = pruned
							bt.logging.info(
								f"[CONTENT] Enforced cap: pruned {pruned} from {source_type} '{source_name}' (count {count} -> ≤ {self.max_per_source})"
							)
		except Exception as e:
			bt.logging.error(f"Error enforcing source caps: {e}")
		return results

	def get_media_entry_by_file_path(self, file_path: str) -> Optional[MediaEntry]:
		return self.content_db.get_media_entry_by_file_path(file_path)

	def delete_media_by_file_path(self, file_path: str) -> bool:
		try:
			# Delete from filesystem
			file_path_obj = Path(file_path)
			if file_path_obj.exists():
				success = self.media_storage.delete_media_file(file_path_obj)
				if not success:
					return False

			# Delete from database
			return self.content_db.delete_media_entry_by_file_path(file_path)

		except Exception as e:
			bt.logging.error(f"Error deleting media {file_path}: {e}")
			return False

	def get_stats(self) -> Dict[str, Any]:
		return self.content_db.get_stats()

	def get_source_stats(self) -> Dict[str, Dict[str, int]]:
		"""
		Returns:
			Dictionary with counts for each source type and source name
		"""
		return self.content_db.get_source_counts()

	def get_dataset_media_counts(self) -> Dict[str, int]:
		"""
		Get counts of dataset media entries (NULL prompt_id) by modality and media type.

		Returns:
			Dictionary with counts for each modality/media_type combination
		"""
		return self.content_db.get_dataset_media_counts()

	def get_sources_needing_data(self) -> Dict[str, List[str]]:
		"""
		Returns:
			Dictionary with source types as keys and lists of source names that need data
		"""
		source_stats = self.content_db.get_source_counts()
		sources_needing_data: Dict[str, List[str]] = {}
		for source_type, sources in source_stats.items():
			sources_needing_data[source_type] = []
			for source_name, count in sources.items():
				if count < self.min_source_threshold:
					sources_needing_data[source_type].append(source_name)
		return sources_needing_data

	def get_source_count(self, source_type: str, source_name: str) -> int:
		"""
		Args:
			source_type: 'dataset', 'scraper', or 'generated'
			source_name: Name of the dataset, scraper, or model

		Returns:
			Count of media items for this source
		"""
		return self.content_db.get_source_count(source_type, source_name)

	def needs_more_data(self, source_type: str, source_name: str) -> bool:
		"""
		Args:
			source_type: 'dataset', 'scraper', or 'generated'
			source_name: Name of the source

		Returns:
			True if the source needs more data
		"""
		return self.get_source_count(source_type, source_name) < self.min_source_threshold

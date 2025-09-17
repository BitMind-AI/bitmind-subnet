import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import bittensor as bt
import numpy as np

from gas.cache.content_db import ContentDB, SOURCE_TYPE_TO_DB_NAME_FIELD
from gas.cache.media_storage import MediaStorage
from gas.cache.types import Media, MediaEntry, PromptEntry
from gas.cache.util import extract_media_info, get_format_from_content
from gas.types import MediaType, Modality, SourceType, SOURCE_TYPE_TO_NAME
from gas.utils.huggingface_uploads import upload_images_to_hf, upload_videos_to_hf


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

			media_id = self.content_db.add_media_entry(
				prompt_id=media_data.prompt_id,
				file_path=save_path,
				modality=media_data.modality,
				media_type=media_data.media_type,
				source_type=SourceType.GENERATED,
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

	def write_miner_media(
		self,
		modality: Modality,
		media_type: MediaType,
		prompt_id: str,
		uid: int,
		hotkey: str,
		media_content: bytes,
		content_type: str,
		task_id: str,
		model_name: Optional[str] = None,
	) -> Optional[str]:
		"""
		Write miner-generated binary media to storage.
		Follows the same pattern as other write methods.

		Args:
			uid: Miner UID
			hotkey: Miner hotkey
			binary_data: Raw binary media data
			content_type: MIME content type (e.g., "image/png", "video/mp4")
			task_id: Unique task identifier for filename
			model_name: Optional model name

		Returns:
			Path to saved file if successful, None if failed
		"""
		try:
			media_data = Media(
				modality=modality,
				media_type=media_type,
				prompt_id=prompt_id,
				media_content=media_content,
				format=get_format_from_content(media_content, modality),
				model_name=model_name,
				metadata={"uid": uid, "task_id": task_id, "source": "miner"}
			)

			save_path, mask_path = self.media_storage.write_media(media_data)
			if save_path is None:
				bt.logging.error("Failed to write miner media to filesystem")
				return None

			resolution, file_size = extract_media_info(save_path, modality)
			media_id = self.content_db.add_media_entry(
				prompt_id=prompt_id,
				file_path=save_path,
				modality=modality,
				media_type=media_type,
				source_type=SourceType.MINER,
				uid=uid,
				hotkey=hotkey,
				model_name=model_name,
				verified=False,  # Initially unverified
				timestamp=int(time.time()),
				resolution=resolution,
				file_size=file_size,
				format=media_data.format,
			)

			bt.logging.info(f"Saved miner media to {save_path} with database ID: {media_id}")
			return str(save_path)

		except Exception as e:
			bt.logging.error(f"Error writing miner media: {e}")
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

		media_id = self.content_db.add_media_entry(
			prompt_id=media_data.prompt_id,
			file_path=save_path,
			modality=media_data.modality,
			media_type=media_data.media_type,
			source_type=SourceType.SCRAPER,
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
			source_type=SourceType.DATASET,
			dataset_name=dataset_name,
			dataset_source_file=dataset_source_file,
			dataset_index=dataset_index,
			mask_path=mask_path,
			timestamp=int(time.time()),
			resolution=resolution,
			file_size=file_size,
			format=media_data.format,
		)

		return save_path

	def sample_prompts(
        self, k: 
        int = 1, 
        remove: bool = False, 
        strategy: str = "random"
	) -> List[PromptEntry]:
		return self.content_db.sample_prompt_entries(
			k=k, remove=remove, strategy=strategy, content_type="prompt")

	def sample_search_queries(
        self, 
        k: int = 1, 
        remove: bool = False, 
        strategy: str = "random"
	) -> List[PromptEntry]:
		return self.content_db.sample_prompt_entries(
			k=k, remove=remove, strategy=strategy, content_type="search_query")

	def get_prompt_by_id(self, prompt_id: str) -> Optional[str]:
		return self.content_db.get_prompt_by_id(prompt_id)

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
			remove=False,
		)
		if not media_entries:
			print(f"No media available in database for {modality}/{media_type}")
			return {'count': 0, 'items': []}

		# Split by source_type so we only remove dataset/scraper on sample
		generated_entries: List[MediaEntry] = [
			e for e in media_entries 
			if getattr(e, 'source_type', None) == SourceType.GENERATED
		]
		non_generated_entries: List[MediaEntry] = [
			e for e in media_entries
			if getattr(e, 'source_type', None) != SourceType.GENERATED
		]

		items: List[Dict[str, Any]] = []
		if non_generated_entries:
			non_gen_items = self.media_storage.retrieve_media(
				media_entries=non_generated_entries,
				modality=modality,
				remove_from_cache=False,
				**kwargs,
			)['items']
			items.extend(non_gen_items)

			if should_remove:
				for entry in non_generated_entries:
					self.delete_media(file_path=entry.file_path)

		if generated_entries:
			gen_items = self.media_storage.retrieve_media(
				media_entries=generated_entries,
				modality=modality,
				remove_from_cache=False,
				**kwargs,
			)['items']
			items.extend(gen_items)

		combined_entries: List[MediaEntry] = non_generated_entries + generated_entries
		for media, db_entry in zip(items, combined_entries):
			media['id'] = db_entry.id
			media['metadata'] = db_entry.to_dict()

			origin_field = SOURCE_TYPE_TO_NAME[db_entry.source_type]
			origin_value = getattr(db_entry, origin_field)
			media['source_type'] = db_entry.source_type
			media['source_name'] = origin_value

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

	def _check_and_prune_source(self, source_type: SourceType, source_name: str, max_count: int) -> int:
		if not self.enable_source_limits:
			return 0
		current_count = self.content_db.get_source_count(source_type, source_name)
		if current_count >= max_count:
			pruned = self.content_db.prune_source_media(
				source_type, source_name, max_count, self.prune_strategy
			)
			if pruned > 0:
				bt.logging.info(f"Pruned {pruned} items from {source_type.value} '{source_name}' to enforce cap {max_count}")
			return pruned
		return 0

	def enforce_source_caps(self) -> Dict[str, int]:
		results: Dict[str, int] = {}
		if not self.enable_source_limits:
			return results
		try:
			counts = self.content_db.get_source_counts()
			for source_type_str, sources in counts.items():
				st = SourceType(source_type_str)
				for source_name, count in sources.items():
					if count > self.max_per_source:
						pruned = self._prune_source_media(st, source_name, self.max_per_source)
						if pruned > 0:
							key = f"{st.value}:{source_name}"
							results[key] = pruned
							bt.logging.info(
								f"[CONTENT] Enforced cap: pruned {pruned} from {st.value} '{source_name}' "
								f"(count {count} -> ≤ {self.max_per_source})"
							)
		except Exception as e:
			bt.logging.error(f"Error enforcing source caps: {e}")
		return results

	def _prune_source_media(self, source_type: SourceType, source_name: str, max_count: int) -> int:

		col = SOURCE_TYPE_TO_DB_NAME_FIELD.get(source_type)
		if not col:
			return 0

		current = self.content_db.get_source_count(source_type, source_name)
		if current <= max_count:
			return 0

		to_remove = current - max_count
		order_clause = 'created_at ASC' if self.prune_strategy in ('oldest', 'least_used') else 'RANDOM()'

		with self.content_db._get_db_connection() as conn:
			cursor = conn.execute(
				f"""
				SELECT file_path FROM media
				WHERE source_type = ? AND {col} = ?
				ORDER BY {order_clause}
				LIMIT ?
				""",
				(source_type.value, source_name, to_remove),
			)
			file_paths = [row[0] for row in cursor.fetchall()]

		deleted_count = 0
		for file_path in file_paths:
			if self.delete_media(file_path=file_path):
				deleted_count += 1

		return deleted_count

	def get_media_entry_by_file_path(self, file_path: str) -> Optional[MediaEntry]:
		return self.content_db.get_media_entry_by_file_path(file_path)

	def delete_media(self, file_path: str = None, media_id: str = None) -> bool:
		if not file_path and not media_id:
			return False

		try:
			if media_id and not file_path:
				media_entry = self.content_db.get_media_entries(media_id=media_id)
				if not media_entry:
					return False
				file_path = media_entry[0].file_path

			file_path_obj = Path(file_path)
			if file_path_obj.exists():
				success = self.media_storage.delete_media_file(file_path_obj)
				if not success:
					return False

			return self.content_db.delete_media_entry_by_file_path(file_path)

		except Exception as e:
			bt.logging.error(f"Error deleting media: {e}")
			return False

	def delete_media_by_file_path(self, file_path: str) -> bool:
		return self.delete_media(file_path=file_path)

	def get_miner_media(self, verification_status: Optional[str] = None) -> List[MediaEntry]:
		"""
		Get miner media by verification status.
		
		Args:
			verification_status: Optional verification status filter:
				- 'pending': Not verified yet (verified=0, failed_verification=0)
				- 'verified': Passed verification (verified=1, failed_verification=0)  
				- 'failed': Failed verification (verified=0, failed_verification=1)
				- None: Return all miner media regardless of verification status
		
		Returns:
			List of MediaEntry objects matching the verification status
		"""
		return self.content_db.get_miner_media(verification_status=verification_status)

	def mark_miner_media_verified(self, media_id: str) -> bool:
		return self.content_db.mark_miner_media_verified(media_id)

	def mark_miner_media_failed_verification(self, media_id: str) -> bool:
		"""Mark miner media as failed verification."""
		return self.content_db.mark_miner_media_failed_verification(media_id)

	def get_unuploaded_media(
		self, 
		limit: int = 100, 
		modality: str = None, 
		verified_only: bool = False, 
		skip_verified: bool = False
	) -> List[MediaEntry]:
		return self.content_db.get_unuploaded_media(
			limit=limit,
			modality=modality,
			verified_only=verified_only,
			skip_verified=skip_verified
		)

	def mark_media_uploaded(self, media_ids: List[str]) -> bool:
		return self.content_db.mark_media_uploaded(media_ids)

	def mark_media_rewarded(self, media_ids: List[str]) -> bool:
		"""Mark media entries as rewarded."""
		return self.content_db.mark_media_rewarded(media_ids)

	def get_unrewarded_verified_miner_media(self, limit: int = 100) -> List[MediaEntry]:
		"""Get verified miner media entries that haven't been rewarded yet."""
		return self.content_db.get_unrewarded_verified_miner_media(limit=limit)

	def get_unrewarded_verification_stats(self, limit: int = None) -> Dict[str, Dict[str, Any]]:
		"""
		Get verification statistics for unrewarded miner media (pass rates, etc.).
		Returns raw statistics without computing rewards - that's done in rewards.py.

		Args:
			limit: Maximum number of unrewarded entries to consider per miner

		Returns:
			Dict mapping miner hotkey to verification stats:
			{
				"hotkey": {
					"uid": int,
					"total_verified": int,        # Count of verified media
					"total_submissions": int,     # Count of all submissions (verified + failed + pending)
					"total_failed": int,         # Count of failed verification media
					"total_evaluated": int,      # Count of evaluated media (verified + failed)
					"pass_rate": float,          # verified / (verified + failed)
					"media_ids": List[str]       # IDs of verified media to mark as rewarded
				}
			}
		"""
		try:
			verified_media = self.get_unrewarded_verified_miner_media(limit=limit or 1000)
			if not verified_media:
				bt.logging.debug("No unrewarded verified miner media found")
				return {}

			miner_stats = {}
			for media in verified_media:
				if not media.hotkey or not media.uid:
					continue
				
				hotkey = media.hotkey
				if hotkey not in miner_stats:
					miner_stats[hotkey] = {
						"uid": media.uid,
						"verified_media_ids": [],
						"total_verified": 0,
						"total_submissions": 0,
						"total_failed": 0
					}
				
				miner_stats[hotkey]["verified_media_ids"].append(media.id)
				miner_stats[hotkey]["total_verified"] += 1

			# Get total submission counts per miner (verified + failed + pending)
			for hotkey, stats in miner_stats.items():
				uid = stats["uid"]
				
				# Get all miner media for this hotkey/uid
				all_miner_media = self.get_miner_media(verification_status=None)
				miner_media = [m for m in all_miner_media if m.hotkey == hotkey and m.uid == uid]
				
				stats["total_submissions"] = len(miner_media)
				stats["total_failed"] = len([m for m in miner_media if m.failed_verification])

			# Calculate pass rates
			verification_stats = {}
			for hotkey, stats in miner_stats.items():
				verified = stats["total_verified"]
				failed = stats["total_failed"]
				total_evaluated = verified + failed
				
				# pass rate verified media
				if total_evaluated > 0:
					pass_rate = verified / total_evaluated
				else:
					pass_rate = 0.0
				
				verification_stats[hotkey] = {
					"uid": stats["uid"],
					"total_verified": verified,
					"total_submissions": stats["total_submissions"],
					"total_failed": failed,
					"total_evaluated": total_evaluated,
					"pass_rate": pass_rate,
					"media_ids": stats["verified_media_ids"]
				}

			bt.logging.info(f"Retrieved verification stats for {len(verification_stats)} miners")
			return verification_stats

		except Exception as e:
			bt.logging.error(f"Error getting unrewarded verification stats: {e}")
			import traceback
			bt.logging.error(traceback.format_exc())
			return {}

	def upload_batch_to_huggingface(
		self, 
		hf_token: str, 
		hf_dataset_repos: dict, 
		upload_batch_size: int, 
		videos_per_archive: int,
		validator_hotkey: str = None
	):
		"""
		Upload unuploaded media from database to HuggingFace, separated by modality.
		Only uploads verified miner media or validator-generated media.
		"""
		try:			
			#  prioritize verified miner media, then fill with validator media
			media_by_modality = {}
			total_found = 0
			
			for modality in ["image", "video"]:
				# prioritize all verified miner media
				verified_miner_media = self.content_db.get_unuploaded_media(
					limit=None, 
					modality=modality, 
					verified_only=True
				)
				
				# Fill remaining slots with any unuploaded validator media
				remaining_slots = max(0, upload_batch_size - len(verified_miner_media))
				remaining_media = []
				if remaining_slots > 0:
					remaining_media = self.content_db.get_unuploaded_media(
						limit=remaining_slots, 
						modality=modality, 
						skip_verified=True
					)

				unuploaded_media = verified_miner_media + remaining_media
				media_by_modality[modality] = unuploaded_media
				total_found += len(unuploaded_media)

				bt.logging.info(
					f"{modality}: {len(verified_miner_media)} verified miner + "
					f"{len(remaining_media)} validator = {len(unuploaded_media)} total"
				)
			
			if total_found == 0:
				bt.logging.debug("No unuploaded media found in database")
				return

			bt.logging.info(
				f"Found {total_found} unuploaded media files to upload to HuggingFace "
				f"({len(media_by_modality['image'])} images, {len(media_by_modality['video'])} videos)"
			)

			all_successfully_processed_ids = []

			if media_by_modality['image']:
				uploaded_ids = upload_images_to_hf(
					media_entries=media_by_modality['image'],
					hf_token=hf_token,
					dataset_repo=hf_dataset_repos['image'],
					validator_hotkey=validator_hotkey
				)
				all_successfully_processed_ids.extend(uploaded_ids)

			if media_by_modality['video']:
				uploaded_ids = upload_videos_to_hf(
					media_entries=media_by_modality['video'],
					hf_token=hf_token,
					dataset_repo=hf_dataset_repos['video'],
					videos_per_archive=videos_per_archive,
					validator_hotkey=validator_hotkey
				)
				all_successfully_processed_ids.extend(uploaded_ids)

			# Mark all successfully uploaded media as uploaded in db
			if all_successfully_processed_ids:
				success = self.mark_media_uploaded(all_successfully_processed_ids)
				if success:
					bt.logging.info(f"Marked {len(all_successfully_processed_ids)} entries as uploaded in database")
				else:
					bt.logging.warning("Failed to mark media as uploaded in database")

		except Exception as e:
			bt.logging.error(f"Error uploading batch to HuggingFace: {e}")
			import traceback
			bt.logging.error(traceback.format_exc())

	def get_dataset_media_counts(self) -> Dict[str, int]:
		"""
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

	def get_source_count(self, source_type: SourceType, source_name: str) -> int:
		"""
		Args:
			source_type: SourceType.DATASET, SourceType.SCRAPER, or SourceType.GENERATED
			source_name: Name of the dataset, scraper, or model

		Returns:
			Count of media items for this source
		"""
		return self.content_db.get_source_count(source_type, source_name)

	def needs_more_data(self, source_type: SourceType, source_name: str) -> bool:
		"""
		Args:
			source_type: SourceType.DATASET, SourceType.SCRAPER, or SourceType.GENERATED
			source_name: Name of the source

		Returns:
			True if the source needs more data
		"""
		return self.get_source_count(source_type, source_name) < self.min_source_threshold

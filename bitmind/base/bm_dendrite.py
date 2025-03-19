import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Optional, Union, Type, List

import aiohttp
from bittensor_wallet import Keypair, Wallet

from bittensor.core.axon import Axon
from bittensor.core.chain_data import AxonInfo
from bittensor.core.stream import StreamingSynapse
from bittensor.core.synapse import Synapse
from bittensor.utils.btlogging import logging
from bittensor.core.dendrite import Dendrite

class BMDendrite(Dendrite):
    """
    Enhanced Dendrite implementation with improved connection pooling and resilience.
    
    This class extends the standard Dendrite to provide better handling of concurrent
    connections, automatic retries for common network issues, and batch processing
    of multiple axon queries to prevent resource exhaustion.
    
    Args:
        wallet (Optional[Union["Wallet", "Keypair"]]): The wallet or keypair used for
            signing messages. Same as parent Dendrite.
        max_connections (int): Maximum number of total concurrent connections.
        max_connections_per_axon (int): Maximum number of concurrent connections per host.
        retry_attempts (int): Number of retry attempts for recoverable errors.
        batch_size (int): Number of axons to query in a single batch when running async.
        keepalive_timeout (float): How long to keep connections alive in the pool (seconds).
    """
    
    def __init__(
        self, 
        wallet: Optional[Union["Wallet", "Keypair"]] = None,
        max_connections: int = 100,
        max_connections_per_axon: int = 8,
        retry_attempts: int = 2,
        batch_size: int = 20,
        keepalive_timeout: float = 15.0
    ):
        super().__init__(wallet=wallet)
        
        self.max_connections = max_connections
        self.max_connections_per_axon = max_connections_per_axon
        self.retry_attempts = retry_attempts
        self.batch_size = batch_size
        self.keepalive_timeout = keepalive_timeout
        
        self._session = None
        
        self._connection_metrics = {
            "total_requests": 0,
            "retried_requests": 0,
            "failed_requests": 0,
            "successful_requests": 0,
        }
    
    @property
    async def session(self) -> aiohttp.ClientSession:
        """
        An asynchronous property that provides access to the internal aiohttp client session
        with improved connection pooling.
        
        Returns:
            aiohttp.ClientSession: The active aiohttp client session instance with custom connection pooling.
        """
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_axon,
                force_close=False,
                enable_cleanup_closed=True,
                keepalive_timeout=self.keepalive_timeout
            )
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    connect=5.0,
                    sock_connect=5.0,
                    sock_read=10.0
                ),
                raise_for_status=False    # handle HTTP status errors within the class
            )
        return self._session
    
    async def forward(
        self,
        axons: Union[list[Union["AxonInfo", "Axon"]], Union["AxonInfo", "Axon"]],
        synapse: "Synapse" = Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ) -> list[Union["AsyncGenerator[Any, Any]", "Synapse", "StreamingSynapse"]]:
        """
        Enhanced forward method with batch processing and improved error handling.
        
        This implementation processes axons in batches when running asynchronously to prevent
        overwhelming network resources and connection pools.
        
        Args:
            axons: Target axons to query (single axon or list of axons)
            synapse: The Synapse object to send
            timeout: Maximum time to wait for a response
            deserialize: Whether to deserialize the response
            run_async: Whether to run queries concurrently
            streaming: Whether the response is expected as a stream
            
        Returns:
            Response from axons (single response or list of responses)
        """
        is_list = True
        if not isinstance(axons, list):
            is_list = False
            axons = [axons]

        is_streaming_subclass = issubclass(synapse.__class__, StreamingSynapse)
        if streaming != is_streaming_subclass:
            logging.warning(
                f"Argument streaming is {streaming} while issubclass(synapse, StreamingSynapse) is {synapse.__class__.__name__}. This may cause unexpected behavior."
            )
        streaming = is_streaming_subclass or streaming

        async def query_all_axons(
            is_stream: bool,
        ) -> Union["AsyncGenerator[Any, Any]", "Synapse", "StreamingSynapse"]:
            """Query all axons with improved connection handling."""

            async def single_axon_response_with_retry(
                target_axon: Union["AxonInfo", "Axon"],
                retries: int = 0
            ) -> Union["AsyncGenerator[Any, Any]", "Synapse", "StreamingSynapse"]:
                """Process a single axon with retry logic for connection errors."""
                self._connection_metrics["total_requests"] += 1
                try:
                    if is_stream:
                        # If in streaming mode, return the async_generator
                        result = self.call_stream(
                            target_axon=target_axon,
                            synapse=synapse.model_copy(),  # type: ignore
                            timeout=timeout,
                            deserialize=deserialize,
                        )
                        self._connection_metrics["successful_requests"] += 1
                        return result
                    else:
                        # If not in streaming mode, simply call the axon and get the response.
                        result = await self.call(
                            target_axon=target_axon,
                            synapse=synapse.model_copy(),  # type: ignore
                            timeout=timeout,
                            deserialize=deserialize,
                        )
                        self._connection_metrics["successful_requests"] += 1
                        return result
                except (aiohttp.ClientOSError, ConnectionResetError, aiohttp.ServerDisconnectedError) as e:
                    # Retry on common network/connection errors
                    error_str = str(e)
                    is_retryable = (
                        "Broken pipe" in error_str or 
                        "Connection reset" in error_str or
                        "Server disconnected" in error_str
                    )
                    
                    if retries < self.retry_attempts and is_retryable:
                        backoff_time = 0.1 * (2 ** retries)
                        logging.debug(
                            f"Connection error to {target_axon.ip}:{target_axon.port}, "
                            f"retrying in {backoff_time:.2f}s ({retries+1}/{self.retry_attempts})"
                        )
                        self._connection_metrics["retried_requests"] += 1
                        await asyncio.sleep(backoff_time)
                        return await single_axon_response_with_retry(target_axon, retries + 1)
                    
                    self._connection_metrics["failed_requests"] += 1
                    raise

            if not run_async:
                return [
                    await single_axon_response_with_retry(target_axon) for target_axon in axons
                ]
            
            all_responses = []
            for i in range(0, len(axons), self.batch_size):
                batch = axons[i:i+self.batch_size]
                batch_responses = await asyncio.gather(
                    *(single_axon_response_with_retry(target_axon) for target_axon in batch),
                    return_exceptions=True  # Don't let one failure block others
                )
                
                # Process any exceptions that were captured
                for j, response in enumerate(batch_responses):
                    if isinstance(response, Exception):
                        failed_synapse = synapse.model_copy()
                        target_axon = batch[j]
                        failed_synapse = self.preprocess_synapse_for_request(
                            target_axon, failed_synapse, timeout
                        )
                        failed_synapse = self.process_error_message(
                            failed_synapse, 
                            failed_synapse.__class__.__name__, 
                            response
                        )
                        batch_responses[j] = failed_synapse
                
                all_responses.extend(batch_responses)
            
            return all_responses

        responses = await query_all_axons(streaming)
        return responses[0] if len(responses) == 1 and not is_list else responses

    async def call(
        self,
        target_axon: Union["AxonInfo", "Axon"],
        synapse: "Synapse" = Synapse(),
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> "Synapse":
        """
        Enhanced call method with improved error handling for connection issues.
        
        Args:
            target_axon: The target axon to query
            synapse: The Synapse object to send
            timeout: Maximum time to wait for a response
            deserialize: Whether to deserialize the response
            
        Returns:
            The response Synapse object
        """
        
        start_time = time.time()
        target_axon = (
            target_axon.info() if isinstance(target_axon, Axon) else target_axon
        )

        request_name = synapse.__class__.__name__
        url = self._get_endpoint_url(target_axon, request_name=request_name)

        synapse = self.preprocess_synapse_for_request(target_axon, synapse, timeout)

        try:
            self._log_outgoing_request(synapse)

            try:
                async with (await self.session).post(
                    url=url,
                    headers=synapse.to_headers(),
                    json=synapse.model_dump(),
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    json_response = await response.json()
                    self.process_server_response(response, json_response, synapse)
            except aiohttp.ClientPayloadError as e:
                if "Response payload is not completed" in str(e):
                    synapse.dendrite.status_code = "499"
                    synapse.dendrite.status_message = f"Incomplete response payload: {str(e)}"
                else:
                    raise
            except aiohttp.ClientOSError as e:
                if "Broken pipe" in str(e):
                    synapse.dendrite.status_code = "503"
                    synapse.dendrite.status_message = f"Connection broken: {str(e)}"
                else:
                    raise

            synapse.dendrite.process_time = str(time.time() - start_time)

        except Exception as e:
            synapse = self.process_error_message(synapse, request_name, e)

        finally:
            self._log_incoming_response(synapse)
            self.synapse_history.append(Synapse.from_headers(synapse.to_headers()))
            return synapse.deserialize() if deserialize else synapse
    
    async def call_stream(
        self,
        target_axon: Union["AxonInfo", "Axon"],
        synapse: "StreamingSynapse" = Synapse(),
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> "AsyncGenerator[Any, Any]":
        """
        Enhanced call_stream method for streaming responses with improved error handling.
        
        Args:
            target_axon: The target axon to query
            synapse: The Synapse object to send
            timeout: Maximum time to wait for initial response
            deserialize: Whether to deserialize the response
            
        Yields:
            Response chunks from the streaming endpoint
        """
        start_time = time.time()
        target_axon = (
            target_axon.info() if isinstance(target_axon, Axon) else target_axon
        )

        request_name = synapse.__class__.__name__
        endpoint = (
            f"0.0.0.0:{str(target_axon.port)}"
            if target_axon.ip == str(self.external_ip)
            else f"{target_axon.ip}:{str(target_axon.port)}"
        )
        url = f"http://{endpoint}/{request_name}"

        synapse = self.preprocess_synapse_for_request(target_axon, synapse, timeout)

        try:
            self._log_outgoing_request(synapse)
            stream_timeout = aiohttp.ClientTimeout(
                total=None,
                connect=10.0,
                sock_connect=10.0,
                sock_read=timeout
            )

            async with (await self.session).post(
                url,
                headers=synapse.to_headers(),
                json=synapse.model_dump(),
                timeout=stream_timeout,
            ) as response:
                try:
                    async for chunk in synapse.process_streaming_response(response):
                        yield chunk
                except (aiohttp.ClientPayloadError, aiohttp.ClientOSError) as e:
                    error_msg = str(e)
                    if "Broken pipe" in error_msg or "incomplete" in error_msg.lower():
                        logging.warning(f"Streaming interrupted: {error_msg}")
                        # The stream was interrupted, but we might have received partial data, so continue
                
                json_response = synapse.extract_response_json(response)
                self.process_server_response(response, json_response, synapse)

            synapse.dendrite.process_time = str(time.time() - start_time)

        except Exception as e:
            synapse = self.process_error_message(synapse, request_name, e)

        finally:
            self._log_incoming_response(synapse)
            self.synapse_history.append(Synapse.from_headers(synapse.to_headers()))
            if deserialize:
                yield synapse.deserialize()
            else:
                yield synapse
    
    def get_connection_metrics(self) -> dict:
        """
        Get metrics about connection usage and errors.
        
        Returns:
            dict: A dictionary containing connection metrics
        """
        return self._connection_metrics.copy()
    
    def reset_connection_metrics(self) -> None:
        """Reset all connection metrics counters"""
        self._connection_metrics = {
            "total_requests": 0,
            "retried_requests": 0,
            "failed_requests": 0,
            "successful_requests": 0,
        }

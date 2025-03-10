import asyncio
import time
import uuid
import logging
from typing import Optional, Any, List, Union, Type

import aiohttp
from substrateinterface.keypair import Keypair
from bittensor.core.synapse import TerminalInfo
from bittensor.core.axon import Axon
from bittensor.utils import networking

# Set up logging
logger = logging.getLogger("minimal_dendrite")

# Error mapping
DENDRITE_ERROR_MAPPING = {
    aiohttp.ClientConnectorError: ("503", "Service unavailable"),
    asyncio.TimeoutError: ("408", "Request timeout"),
    aiohttp.ClientResponseError: (None, "Client response error"),
    aiohttp.ClientPayloadError: ("400", "Payload error"),
    aiohttp.ClientError: ("500", "Client error"),
    aiohttp.ServerTimeoutError: ("504", "Server timeout error"),
    aiohttp.ServerDisconnectedError: ("503", "Service disconnected"),
    aiohttp.ServerConnectionError: ("503", "Service connection error"),
}
DENDRITE_DEFAULT_ERROR = ("422", "Failed to parse response")

class MinimalDendrite:
    """Minimal Dendrite implementation that works with bittensor synapses"""
    
    def __init__(self, keypair: Optional[Keypair] = None):
        """Initialize with a substrate keypair"""
        self.uuid = str(uuid.uuid1())
        self.external_ip = networking.get_external_ip()
        self.keypair = keypair
        self._session = None
        
        print("keypair")
        print(self.keypair)
        
    @property
    async def session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp client session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
        
    async def aclose_session(self):
        """Close the session asynchronously"""
        if self._session:
            await self._session.close()
            self._session = None
            
    def close_session(self):
        """Close the session synchronously"""
        if self._session:
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._session.close())
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            self._session = None
            
    def __call__(self, *args, **kwargs):
        """Allow dendrite to be called directly"""
        return self.forward(*args, **kwargs)
        
    async def forward(
        self,
        axons,
        synapse=None,
        timeout=12.0,
        deserialize=True,
        run_async=True,
        streaming=False,
    ):
        """Send requests to axons and collate responses"""
        is_list = True
        if not isinstance(axons, list):
            is_list = False
            axons = [axons]
        
        async def query_axons():
            async def process_axon(target_axon):
                try:
                    axon_ip = target_axon.ip if hasattr(target_axon, 'ip') else target_axon.get('ip', 'unknown')
                    axon_port = target_axon.port if hasattr(target_axon, 'port') else target_axon.get('port', 'unknown')
                    axon_hotkey = target_axon.hotkey if hasattr(target_axon, 'hotkey') else target_axon.get('hotkey', 'unknown')
                    logger.info(f"Sending request to axon: {axon_ip}:{axon_port} (hotkey: {axon_hotkey[:10]}...)")
                    
                    return await self.call(
                        target_axon=target_axon,
                        synapse=synapse,
                        timeout=timeout,
                        deserialize=deserialize
                    )
                except Exception as e:
                    axon_ip = target_axon.ip if hasattr(target_axon, 'ip') else target_axon.get('ip', 'unknown')
                    axon_port = target_axon.port if hasattr(target_axon, 'port') else target_axon.get('port', 'unknown')
                    logger.error(f"Error in axon query to {axon_ip}:{axon_port}: {str(e)}")
                    return [-1, -1, -1]  # Error indicator
            
            if not run_async:
                results = []
                for axon in axons:
                    try:
                        results.append(await process_axon(axon))
                    except Exception as e:
                        axon_ip = axon.ip if hasattr(axon, 'ip') else axon.get('ip', 'unknown')
                        axon_port = axon.port if hasattr(axon, 'port') else axon.get('port', 'unknown')
                        logger.warning(f"Error querying axon {axon_ip}:{axon_port}: {e}")
                        results.append([-1, -1, -1])
                return results
            
            # Run concurrently
            tasks = [process_axon(axon) for axon in axons]
            return await asyncio.gather(*tasks, return_exceptions=False)
            
        responses = await query_axons()
        return responses[0] if len(responses) == 1 and not is_list else responses
        
    async def call(
        self,
        target_axon,
        synapse=None,
        timeout=12.0,
        deserialize=True
    ):
        """Send a request to an axon and process the response"""
        start_time = time.time()
        
        # Handle axon info format
        if isinstance(target_axon, Axon):
            target_axon = target_axon.info()
            
        # Get request endpoint
        request_name = synapse.__class__.__name__
        endpoint = (
            f"0.0.0.0:{str(target_axon.port)}"
            if target_axon.ip == str(self.external_ip)
            else f"{target_axon.ip}:{str(target_axon.port)}"
        )
        url = f"http://{endpoint}/{request_name}"
        
        # Prepare the synapse
        synapse.timeout = timeout
        synapse.dendrite = TerminalInfo(
            ip=self.external_ip,
            version=1,  # Simplified version
            nonce=time.time_ns(),
            uuid=self.uuid,
            hotkey=self.keypair.ss58_address if self.keypair else "",
        )
        
        synapse.axon = TerminalInfo(
            ip=target_axon.ip,
            port=target_axon.port,
            hotkey=target_axon.hotkey,
        )
        
        # Sign the request if we have a keypair
        if self.keypair and hasattr(synapse, 'body_hash'):
            message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.dendrite.uuid}.{synapse.body_hash}"
            synapse.dendrite.signature = f"0x{self.keypair.sign(message).hex()}"
            
        logger.info(f"Sending {request_name} request to {target_axon.ip}:{target_axon.port} (hotkey: {target_axon.hotkey[:10]}...)")
        
        try:
            # Make the request
            async with (await self.session).post(
                url=url,
                headers=synapse.to_headers(),
                json=synapse.model_dump(),
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                # Process response
                json_response = await response.json()
                process_time = time.time() - start_time
                
                # For list responses (common with bitmind)
                if isinstance(json_response, list):
                    logger.info(f"Request to {target_axon.ip}:{target_axon.port} successful. "
                               f"Status: {response.status}. Process time: {process_time:.4f}s. "
                               f"Response type: list with {len(json_response)} items")
                    return json_response
                
                # Handle synapse update for object responses
                if response.status == 200:
                    try:
                        server_synapse = synapse.__class__(**json_response)
                        for key in synapse.model_dump().keys():
                            try:
                                setattr(synapse, key, getattr(server_synapse, key))
                            except Exception:
                                pass
                        logger.info(f"Request to {target_axon.ip}:{target_axon.port} successful. "
                                   f"Status: {response.status}. Process time: {process_time:.4f}s. "
                                   f"Response size: {len(str(json_response))} chars")
                    except Exception as e:
                        logger.warning(f"Error updating synapse from {target_axon.ip}:{target_axon.port}: {e}")
                else:
                    synapse.axon.status_code = response.status
                    synapse.axon.status_message = json_response.get("message")
                    logger.warning(f"Non-200 response from {target_axon.ip}:{target_axon.port}. "
                                  f"Status: {response.status}. Message: {json_response.get('message', 'No message')}. "
                                  f"Process time: {process_time:.4f}s")
                
                # Update timing information
                synapse.dendrite.process_time = str(process_time)
                
                # Return deserialized response if requested
                return synapse.deserialize() if deserialize else synapse
                
        except asyncio.TimeoutError:
            process_time = time.time() - start_time
            logger.warning(f"Timeout connecting to {target_axon.ip}:{target_axon.port} after {timeout}s. "
                          f"Elapsed time: {process_time:.4f}s")
            synapse.dendrite.status_code = "408" 
            synapse.dendrite.status_message = f"Request timeout after {timeout} seconds"
            return [-1, -1, -1]
            
        except aiohttp.ClientConnectorError as e:
            process_time = time.time() - start_time
            # This is common and expected for offline miners - debug level only
            logger.debug(f"Cannot connect to host {target_axon.ip}:{target_axon.port} - {str(e)}. "
                        f"Elapsed time: {process_time:.4f}s")
            synapse.dendrite.status_code = "503"
            synapse.dendrite.status_message = f"Service unavailable at {target_axon.ip}:{target_axon.port}"
            return [-1, -1, -1]
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Error in call to {target_axon.ip}:{target_axon.port}: {str(e)}. "
                        f"Request type: {request_name}. Elapsed time: {process_time:.4f}s")
            error_info = DENDRITE_ERROR_MAPPING.get(type(e), DENDRITE_DEFAULT_ERROR)
            status_code, status_message = error_info
            synapse.dendrite.status_code = status_code or "500"
            synapse.dendrite.status_message = f"{status_message}: {str(e)}"
            return [-1, -1, -1]
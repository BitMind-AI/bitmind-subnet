# Generative Mining Guide

## Before You Proceed

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with generative mining operations.

## Generative Mining Overview

Generative miners create synthetic media (images and videos) according to prompts from validators. Miners are rewarded based on their ability to:
- Generate high-quality content that passes validation checks
- Create convincing synthetic media that challenges discriminative miners  
- Respond quickly to generation requests
- Maintain consistent uptime and availability

Generative miners operate as FastAPI servers that receive generation requests from validators and respond asynchronously via webhooks.

## Configuration Setup

### Environment Configuration

Create a `.env.gen_miner` file in the project root to configure your generative miner:

```bash
# ======= Generative Miner Configuration =======

# Wallet Configuration (Required)
BT_WALLET_NAME=your_wallet_name
BT_WALLET_HOTKEY=your_hotkey_name

# Network Configuration
BT_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
BT_NETUID=34

# Axon Configuration
BT_AXON_PORT=8093
BT_AXON_IP=0.0.0.0
BT_AXON_EXTERNAL_IP=auto

# Miner Settings
MINER_DEVICE=auto
MINER_OUTPUT_DIR=/tmp/generated_content
MINER_MAX_CONCURRENT_TASKS=5
MINER_TASK_TIMEOUT=300

# Logging
BT_LOGGING_LEVEL=info

# API Keys (Optional - for 3rd party services)
# Configure API keys for external services, or use local generation
OPENAI_API_KEY=your_openai_api_key
OPEN_ROUTER_API_KEY=your_openrouter_api_key

# Optional Settings  
AUTO_UPDATE=false
MINER_NO_FORCE_VALIDATOR_PERMIT=false
```

### Network Configuration

**Mainnet (SN34)**:
```bash
BT_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
BT_NETUID=34
```

**Testnet (SN379)**:
```bash
BT_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443
BT_NETUID=379
```

## Generation Services

Generative miners support multiple generation approaches. You can choose between external API services or local open source models. The default options are listed below, but miners are free to add any service they want. This is where generative miners can build an edge -- pick models that generate tough-to-classify examples to boost your fool-rates and your incentive. 

### OpenAI Service (DALL-E)
- **API Key**: `OPENAI_API_KEY`
- **Supported**: Image generation
- **Models**: DALL-E 3, DALL-E 2

### OpenRouter Service  
- **API Key**: `OPEN_ROUTER_API_KEY`
- **Supported**: Image generation
- **Models**: Google Gemini Flash Image Preview, various other models
- **Website**: [OpenRouter.ai](https://openrouter.ai)

### Local Service (Open Source Models)
- **API Key**: None required
- **Supported**: Local model inference with open source models
- **Models**: Stable Diffusion, FLUX, and other Hugging Face models
- **Requirements**: GPU with sufficient VRAM (8GB+ recommended)
- **Advantages**: No ongoing API costs, full control over models, privacy
- **Note**: Requires additional model setup and local compute resources

**Important**: Configure at least one generation method (API service OR local models) for your miner to be functional. Local generation with open source models is a cost-effective alternative to paid API services.

## Starting the Miner

### Using gascli (Recommended)

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Start the generative miner:
```bash
# Start the miner
gascli generator start

# Using aliases
gascli gen start
gascli g start
```

### Alternative Method

You can also start the miner using PM2 directly:
```bash
pm2 start gen_miner.config.js
```

## Miner Management

### Status and Monitoring

```bash
# Check miner status
gascli generator status

# View miner logs  
gascli generator logs

# Follow logs in real-time
gascli generator logs --follow

# Show miner configuration and API key status
gascli generator info
```

### Starting and Stopping

```bash
# Start the miner
gascli generator start

# Stop the miner
gascli generator stop

# Restart the miner
gascli generator restart

# Delete the miner process
gascli generator delete
```

## What Happens During Operation

1. **Service Registration**: The miner registers on the Bittensor network with your wallet and hotkey
2. **Service Initialization**: Available generation services are initialized (API services with key validation or local models)
3. **Request Handling**: Validators send generation requests to your miner's FastAPI endpoints
4. **Task Processing**: Requests are queued and processed using configured services (API or local models)
5. **Webhook Responses**: Results are sent back to validators via webhook URLs
6. **Reward Distribution**: Miners are scored based on generation quality and response time

### API Endpoints

Your miner exposes these endpoints for validators:

- `POST /gen_image` - Image generation requests
- `POST /gen_video` - Video generation requests (coming soon)
- `GET /health` - Health check endpoint
- `GET /miner_info` - Miner information and capabilities
- `GET /status/{task_id}` - Task status queries

## Configuration Parameters

### Core Settings

- `BT_WALLET_NAME`: Your Bittensor wallet name
- `BT_WALLET_HOTKEY`: Your wallet hotkey name  
- `BT_NETUID`: Subnet ID (34 for mainnet, 379 for testnet)
- `BT_AXON_PORT`: Port for your miner's API server

### Performance Settings

- `MINER_MAX_CONCURRENT_TASKS`: Maximum parallel generation tasks (default: 5)
- `MINER_TASK_TIMEOUT`: Maximum time per task in seconds (default: 300)
- `MINER_OUTPUT_DIR`: Directory for generated content and logs
- `MINER_DEVICE`: Computing device (`auto`, `cuda`, `cpu`)

## Troubleshooting

### Common Issues

**Miner fails to start**:
- Check wallet configuration in `.env.gen_miner`
- Ensure virtual environment is activated
- Verify network connectivity to chain endpoint

**No generation services available**:
- Check API key configuration with `gascli generator info`
- Verify API keys are valid and have sufficient credits
- Consider using local open source models as an alternative to API services
- Check logs for service initialization errors

**Low rewards**:
- Monitor generation quality and response times
- Ensure stable internet connection and uptime
- Consider upgrading API service plans for better performance

### Getting Help

```bash
# General help
gascli generator --help

# Command-specific help  
gascli generator start --help
gascli generator logs --help

# Check service status
gascli generator info
```

**Note**: Remember to activate the virtual environment with `source .venv/bin/activate` before running any `gascli` commands. 

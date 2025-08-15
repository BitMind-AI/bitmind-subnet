const path = require('path');
const os = require('os');

// Helper functions
function getPythonInterpreter() {
  const projectRoot = __dirname;
  const venvPython = path.join(projectRoot, '.venv', 'bin', 'python');
  const fs = require('fs');
  return fs.existsSync(venvPython) ? venvPython : 'python3';
}

function getNetworkSettings(chainEndpoint) {
  if (chainEndpoint.includes('test')) return 379;
  if (chainEndpoint.includes('finney')) return 34;
  return null;
}

function getLogParam(loglevel) {
  switch (loglevel) {
    case 'trace': return '--logging.trace';
    case 'debug': return '--logging.debug';
    default: return '--logging.info';
  }
}

function getAutoUpdateParam(autoUpdate) {
  return autoUpdate === 'true' ? '' : '--autoupdate-off';
}

function getHeartbeatParam(heartbeat) {
  return heartbeat === 'true' ? '--heartbeat' : '';
}

// Load environment variables
require('dotenv').config({ path: path.resolve(__dirname, '.env.validator') });

// Get configuration from environment with defaults
const config = {
  // Wallet
  walletName: process.env.WALLET_NAME || 'default',
  walletHotkey: process.env.WALLET_HOTKEY || 'default',
  
  // Network
  chainEndpoint: process.env.CHAIN_ENDPOINT || '',
  proxyPort: process.env.PROXY_PORT || '10913',
  
  // Cache
  cacheDir: process.env.SN34_CACHE_DIR || path.join(os.homedir(), '.cache', 'sn34'),
  
  // Device
  device: process.env.DEVICE || 'cuda',
  
  // Logging
  loglevel: process.env.LOGLEVEL || 'info',
  
  // Features
  autoUpdate: process.env.AUTO_UPDATE || 'false',
  heartbeat: process.env.HEARTBEAT || 'false',
  
  // Service intervals
  scraperInterval: process.env.SCRAPER_INTERVAL || '300',
  datasetInterval: process.env.DATASET_INTERVAL || '1800',
  
  // Service selection
  startValidator: process.env.START_VALIDATOR !== 'false',
  startGenerator: process.env.START_GENERATOR !== 'false',
  startData: process.env.START_DATA !== 'false',
};

// Determine netuid
const netuid = getNetworkSettings(config.chainEndpoint);

// Build command parameters
const logParam = getLogParam(config.loglevel);
const autoUpdateParam = getAutoUpdateParam(config.autoUpdate);
const heartbeatParam = getHeartbeatParam(config.heartbeat);
const pythonInterpreter = getPythonInterpreter();

// Project paths
const projectRoot = __dirname;
const validatorScript = path.join(projectRoot, 'neurons', 'validator', 'validator.py');
const generatorScript = path.join(projectRoot, 'gas', 'services', 'generator_service.py');
const dataScript = path.join(projectRoot, 'gas', 'services', 'data_service.py');

// Build apps array
const apps = [];

// Common HF env
const HF_ENV = {
  TRANSFORMERS_VERBOSITY: 'error',
  DIFFUSERS_VERBOSITY: 'error',
  TOKENIZERS_PARALLELISM: 'false',
  HF_HUB_VERBOSITY: 'error',
  ACCELERATE_LOG_LEVEL: 'error',
  HUGGINGFACE_HUB_TOKEN: process.env.HUGGINGFACE_HUB_TOKEN || process.env.HF_TOKEN,
  HF_HOME: path.join(os.homedir(), '.cache', 'huggingface'),
  HF_HUB_DISABLE_TELEMETRY: '1',
};

// Validator service
if (config.startValidator) {
  const validatorArgs = [
    '--wallet.name', config.walletName,
    '--wallet.hotkey', config.walletHotkey,
    '--netuid', netuid.toString(),
    '--subtensor.chain_endpoint', config.chainEndpoint,
    '--proxy.port', config.proxyPort,
    '--neuron.sample-size', '256',
    '--cache.base-dir', config.cacheDir,
    logParam,
    autoUpdateParam,
  ];
  
  if (heartbeatParam) {
    validatorArgs.push(heartbeatParam);
  }
  
  apps.push({
    name: 'sn34-validator',
    script: validatorScript,
    interpreter: pythonInterpreter,
    args: validatorArgs.join(' '),
    env: {
      WANDB_API_KEY: process.env.WANDB_API_KEY,
      ...HF_ENV,
    },
    watch: false,
    instances: 1,
    autorestart: true,
  });
}

// Generator service
if (config.startGenerator) {
  apps.push({
    name: 'sn34-generator',
    script: generatorScript,
    interpreter: pythonInterpreter,
    args: [
      '--cache-dir', config.cacheDir,
      '--device', config.device,
      '--batch-size', '1',
      '--log-level', config.loglevel,
    ].join(' '),
    env: {
      ...HF_ENV,
    },
    watch: false,
    instances: 1,
    autorestart: true,
  });
}

// Data service
if (config.startData) {
  apps.push({
    name: 'sn34-data',
    script: dataScript,
    interpreter: pythonInterpreter,
    args: [
      '--cache-dir', config.cacheDir,
      '--chain-endpoint', config.chainEndpoint,
      '--scraper-interval', config.scraperInterval,
      '--dataset-interval', config.datasetInterval,
      '--log-level', config.loglevel,
    ].join(' '),
    env: {
      ...HF_ENV,
    },
    watch: false,
    instances: 1,
    autorestart: true,
  });
}

module.exports = {
  apps,
};

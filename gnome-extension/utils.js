import Gio from 'gi://Gio'
import GLib from 'gi://GLib'
import Soup from 'gi://Soup'

Gio._promisify(Gio.Subprocess.prototype, 'communicate_utf8_async')

const LOG_PREFIX = '[whisper-npu]'

export function logDebug (message) {
  console.log(`${LOG_PREFIX} ${message}`) // eslint-disable-line no-undef
}

export async function execCommand (argv, input = null, cancellable = null) {
  let cancelId = 0
  let flags = Gio.SubprocessFlags.STDOUT_PIPE |
              Gio.SubprocessFlags.STDERR_PIPE

  if (input !== null) { flags |= Gio.SubprocessFlags.STDIN_PIPE }

  const proc = new Gio.Subprocess({ argv, flags })
  proc.init(cancellable)

  if (cancellable instanceof Gio.Cancellable) {
    cancelId = cancellable.connect(() => proc.force_exit())
  }

  try {
    const [stdout, stderr] = await proc.communicate_utf8_async(input, null)
    const status = proc.get_exit_status()

    if (status !== 0) {
      throw new Error(stderr ? stderr.trim() : `Command '${argv}' failed with exit code ${status}`)
    }

    return stdout ? stdout.trim() : ''
  } finally {
    if (cancelId > 0) { cancellable.disconnect(cancelId) }
  }
}

export class WhisperClient {
  constructor (host, port) {
    this._session = new Soup.Session()
    this._host = host
    this._port = port
  }

  _uri (path) {
    return GLib.Uri.parse(`http://${this._host}:${this._port}${path}`, GLib.UriFlags.NONE)
  }

  async _request (method, path, body = null) {
    const message = new Soup.Message({ method, uri: this._uri(path) })

    if (body) {
      const bytes = new GLib.Bytes(JSON.stringify(body))
      message.set_request_body_from_bytes('application/json', bytes)
    }

    try {
      const bytes = await this._sendAsync(message)
      if (message.get_status() !== Soup.Status.OK) {
        throw new Error(`HTTP ${message.get_status()}`)
      }
      const text = new TextDecoder().decode(bytes.get_data())
      return JSON.parse(text)
    } catch (e) {
      logDebug(`Request failed: ${method} ${path}: ${e.message}`)
      return null
    }
  }

  _sendAsync (message) {
    return new Promise((resolve, reject) => {
      this._session.send_and_read_async(
        message, GLib.PRIORITY_DEFAULT, null,
        (session, result) => {
          try {
            resolve(session.send_and_read_finish(result))
          } catch (e) {
            reject(e)
          }
        }
      )
    })
  }

  async getHealth () {
    return this._request('GET', '/health')
  }

  async getModels () {
    return this._request('GET', '/models')
  }

  async setDefaultModel (modelName) {
    return this._request('PUT', '/model/default', { model: modelName })
  }

  async rewrite (text, tones) {
    return this._request('POST', '/rewrite', { text, tones })
  }

  async getLlmModels () {
    return this._request('GET', '/llm/models')
  }

  async setLlmModel (modelName) {
    return this._request('PUT', '/llm/model', { model: modelName })
  }

  destroy () {
    this._session = null
  }
}

export class HuggingFaceClient {
  constructor () {
    this._session = new Soup.Session()
  }

  async searchModels (org, query = 'whisper') {
    const url = `https://huggingface.co/api/models?author=${org}&search=${query}&sort=downloads&direction=-1&limit=50`
    const message = new Soup.Message({
      method: 'GET',
      uri: GLib.Uri.parse(url, GLib.UriFlags.NONE)
    })

    try {
      const bytes = await this._sendAsync(message)
      if (message.get_status() !== Soup.Status.OK) {
        throw new Error(`HuggingFace API returned ${message.get_status()}`)
      }
      const text = new TextDecoder().decode(bytes.get_data())
      const models = JSON.parse(text)
      return models
        .filter(m => m.id.toLowerCase().includes('whisper'))
        .map(m => ({
          id: m.id,
          name: m.id.split('/').pop(),
          downloads: m.downloads || 0,
          lastModified: m.lastModified || ''
        }))
    } catch (e) {
      logDebug(`HuggingFace search failed: ${e.message}`)
      return []
    }
  }

  _sendAsync (message) {
    return new Promise((resolve, reject) => {
      this._session.send_and_read_async(
        message, GLib.PRIORITY_DEFAULT, null,
        (session, result) => {
          try {
            resolve(session.send_and_read_finish(result))
          } catch (e) {
            reject(e)
          }
        }
      )
    })
  }

  destroy () {
    this._session = null
  }
}

export function getModelsDir () {
  return GLib.build_filenamev([GLib.get_home_dir(), '.whisper', 'models'])
}

export function listLocalModels () {
  const modelsDir = getModelsDir()
  const dir = Gio.file_new_for_path(modelsDir)

  let enumerator
  try {
    enumerator = dir.enumerate_children('standard::*', 0, null)
  } catch (e) {
    logDebug(`Failed to read models directory: ${e.message}`)
    return []
  }

  const models = []
  let entry
  while ((entry = enumerator.next_file(null))) {
    const name = entry.get_name()
    if (!name.startsWith('.') && entry.get_file_type() === Gio.FileType.DIRECTORY) {
      models.push(name)
    }
  }

  return models.sort()
}

export async function downloadModel (org, modelName, cancellable = null) {
  const dest = GLib.build_filenamev([getModelsDir(), modelName])

  if (GLib.file_test(dest, GLib.FileTest.IS_DIR)) {
    throw new Error(`Model ${modelName} already exists`)
  }

  const url = `https://huggingface.co/${org}/${modelName}`
  return execCommand(
    ['git', 'clone', url, dest],
    null,
    cancellable
  )
}

export function getLlmModelsDir () {
  return GLib.build_filenamev([GLib.get_home_dir(), '.whisper', 'llm-models'])
}

export function listLocalLlmModels () {
  const modelsDir = getLlmModelsDir()
  const dir = Gio.file_new_for_path(modelsDir)

  let enumerator
  try {
    enumerator = dir.enumerate_children('standard::*', 0, null)
  } catch (e) {
    return []
  }

  const models = []
  let entry
  while ((entry = enumerator.next_file(null))) {
    const name = entry.get_name()
    if (!name.startsWith('.') && entry.get_file_type() === Gio.FileType.DIRECTORY) {
      models.push(name)
    }
  }

  return models.sort()
}

export async function downloadLlmModel (org, modelName, cancellable = null) {
  const dest = GLib.build_filenamev([getLlmModelsDir(), modelName])

  if (GLib.file_test(dest, GLib.FileTest.IS_DIR)) {
    throw new Error(`LLM model ${modelName} already exists`)
  }

  const parentDir = Gio.file_new_for_path(getLlmModelsDir())
  try {
    parentDir.make_directory_with_parents(null)
  } catch (e) {
    if (!e.matches(Gio.IOErrorEnum, Gio.IOErrorEnum.EXISTS)) { throw e }
  }

  const url = `https://huggingface.co/${org}/${modelName}`
  return execCommand(['git', 'clone', url, dest], null, cancellable)
}

export async function typeText (text, delayMs = 4) {
  return execCommand(['ydotool', 'type', '-d', String(delayMs), '--', text])
}

export async function backspaceN (n) {
  const promises = []
  for (let i = 0; i < n; i++) {
    promises.push(execCommand(['ydotool', 'key', '14:1', '14:0']))
  }
  return Promise.all(promises)
}

export async function restartService (serviceName) {
  return execCommand(['systemctl', '--user', 'restart', serviceName])
}

export async function writeServiceOverride (serviceName, envVars) {
  const overrideDir = GLib.build_filenamev([
    GLib.get_home_dir(), '.config', 'systemd', 'user',
    `${serviceName}.d`
  ])
  const overridePath = GLib.build_filenamev([overrideDir, 'override.conf'])

  const dir = Gio.file_new_for_path(overrideDir)
  try {
    dir.make_directory_with_parents(null)
  } catch (e) {
    if (!e.matches(Gio.IOErrorEnum, Gio.IOErrorEnum.EXISTS)) { throw e }
  }

  let content = '[Service]\n'
  for (const [key, value] of Object.entries(envVars)) {
    content += `Environment="${key}=${value}"\n`
  }

  const file = Gio.file_new_for_path(overridePath)
  file.replace_contents(content, null, false, Gio.FileCreateFlags.REPLACE_DESTINATION, null)

  await execCommand(['systemctl', '--user', 'daemon-reload'])
}

import GLib from 'gi://GLib'
import Gio from 'gi://Gio'
import GObject from 'gi://GObject'
import St from 'gi://St'
import Clutter from 'gi://Clutter'

import { Extension, gettext as _ } from 'resource:///org/gnome/shell/extensions/extension.js'
import * as Main from 'resource:///org/gnome/shell/ui/main.js'
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js'
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js'

import {
  WhisperClient,
  HuggingFaceClient,
  listLocalModels,
  listLocalLlmModels,
  downloadModel,
  downloadLlmModel,
  writeServiceOverride,
  restartService,
  typeText,
  backspaceN,
  logDebug
} from './utils.js'

const DEVICES = ['NPU', 'CPU', 'GPU']
const BACKENDS = ['openvino', 'whisper-cpp']
const HOTKEYS = [
  'KEY_RIGHTCTRL',
  'KEY_RIGHTALT',
  'KEY_RIGHTSHIFT',
  'KEY_SCROLLLOCK',
  'KEY_PAUSE'
]
const DEFAULT_TONES = ['diplomatic', 'professional']

const DBUS_IFACE = `
<node>
  <interface name="com.whisper.LanguageBuddy">
    <method name="HandleTranscription">
      <arg type="s" direction="in" name="text"/>
      <arg type="b" direction="out" name="handled"/>
    </method>
  </interface>
</node>`

const OVERLAY_PADDING = 100

// ---------------------------------------------------------------------------
// Language Buddy overlay — floating card list at bottom-right
// ---------------------------------------------------------------------------

const LanguageBuddyOverlay = GObject.registerClass(
  class LanguageBuddyOverlay extends St.BoxLayout {
    _init () {
      super._init({
        vertical: true,
        style_class: 'whisper-overlay',
        reactive: true,
        visible: false
      })

      this._timeoutId = null
    }

    show (variants, onSelect, timeoutSec) {
      this.destroy_all_children()

      const header = new St.Label({
        text: _('Language Buddy'),
        style_class: 'whisper-overlay-header'
      })
      this.add_child(header)

      for (const variant of variants) {
        const card = new St.Button({
          style_class: 'whisper-overlay-card',
          x_expand: true,
          reactive: true,
          can_focus: false,
          track_hover: true
        })

        const cardBox = new St.BoxLayout({ vertical: true })

        const toneLabel = new St.Label({
          text: variant.tone.charAt(0).toUpperCase() + variant.tone.slice(1),
          style_class: 'whisper-overlay-tone'
        })
        cardBox.add_child(toneLabel)

        const textLabel = new St.Label({
          text: variant.text,
          style_class: 'whisper-overlay-text'
        })
        textLabel.clutter_text.set_line_wrap(true)
        textLabel.clutter_text.set_ellipsize(0)
        cardBox.add_child(textLabel)

        card.set_child(cardBox)

        card.connect('clicked', () => {
          this._dismiss()
          onSelect(variant.text)
        })

        this.add_child(card)
      }

      this._position()
      this.visible = true

      if (timeoutSec > 0) {
        this._timeoutId = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, timeoutSec, () => {
          this._dismiss()
          return GLib.SOURCE_REMOVE
        })
      }
    }

    _position () {
      if (this._allocId) return
      this._allocId = this.connect('notify::allocation', () => {
        this.disconnect(this._allocId)
        this._allocId = null

        const monitor = Main.layoutManager.primaryMonitor
        if (!monitor) return
        const [, natW] = this.get_preferred_width(-1)
        const [, natH] = this.get_preferred_height(-1)
        this.set_position(
          monitor.x + monitor.width - natW - OVERLAY_PADDING,
          monitor.y + monitor.height - natH - OVERLAY_PADDING
        )
      })
    }

    _dismiss () {
      if (this._allocId) {
        this.disconnect(this._allocId)
        this._allocId = null
      }
      if (this._timeoutId) {
        GLib.source_remove(this._timeoutId)
        this._timeoutId = null
      }
      this.visible = false
      this.destroy_all_children()
    }

    destroy () {
      this._dismiss()
      super.destroy()
    }
  }
)

// ---------------------------------------------------------------------------
// Panel indicator with dropdown menu
// ---------------------------------------------------------------------------

const WhisperIndicator = GObject.registerClass(
  class WhisperIndicator extends PanelMenu.Button {
    _init (extension) {
      super._init(0.0, _('Whisper NPU'))

      this._extension = extension
      this._settings = extension.getSettings()
      this._connected = false
      this._currentModel = null
      this._downloadCancellable = null
      this._cachedRemoteModels = []
      this._cacheReady = false
      this._currentLlmModel = null

      this._client = new WhisperClient(
        this._settings.get_string('server-host'),
        this._settings.get_int('server-port')
      )
      this._hfClient = new HuggingFaceClient()

      const iconPath = GLib.build_filenamev([extension.path, 'icons', 'whisper-npu-symbolic.svg'])
      const gicon = Gio.icon_new_for_string(iconPath)
      const icon = new St.Icon({
        gicon,
        style_class: 'system-status-icon'
      })
      this.add_child(icon)

      this._buildMenu()

      this.menu.connect('open-state-changed', (_menu, isOpen) => {
        if (isOpen) this._refreshStatus()
      })

      this._settingsChangedId = this._settings.connect('changed', (settings, key) => {
        if (key === 'server-host' || key === 'server-port') {
          this._client.destroy()
          this._client = new WhisperClient(
            settings.get_string('server-host'),
            settings.get_int('server-port')
          )
        }
      })

      // Overlay
      this._overlay = new LanguageBuddyOverlay()
      Main.layoutManager.addChrome(this._overlay, {
        affectsStruts: false,
        trackFullscreen: false
      })

      // D-Bus
      this._dbusImpl = Gio.DBusExportedObject.wrapJSObject(DBUS_IFACE, this)
      this._dbusImpl.export(Gio.DBus.session, '/com/whisper/LanguageBuddy')

      this._preloadCache()
    }

    async _preloadCache () {
      try {
        const org = this._settings.get_string('hf-org')
        const [health, remoteModels, llmInfo] = await Promise.all([
          this._client.getHealth(),
          this._hfClient.searchModels(org),
          this._client.getLlmModels()
        ])

        if (health) {
          this._connected = true
          this._currentModel = health.model
        }

        this._currentLlmModel = llmInfo ? llmInfo.current : null

        this._cachedRemoteModels = remoteModels || []
        this._cacheReady = true
        logDebug(`Preloaded ${this._cachedRemoteModels.length} remote models`)

        this._populateDownloadSection()
        this._populateModelSection()
        this._populateLlmModelSection()
      } catch (e) {
        logDebug(`Preload failed: ${e.message}`)
        this._cacheReady = true
      }
    }

    // -- D-Bus method handler -----------------------------------------------

    HandleTranscription (text) {
      const enabled = this._settings.get_boolean('language-buddy-enabled')
      if (!enabled) return false

      logDebug(`Language Buddy received: ${text.substring(0, 80)}`)
      this._processTranscription(text)
      return true
    }

    async _processTranscription (text) {
      const bypass = this._settings.get_boolean('language-buddy-bypass')

      if (bypass) {
        typeText(text).catch(e => logDebug(`typeText failed: ${e.message}`))
      }

      const tones = DEFAULT_TONES
      const result = await this._client.rewrite(text, tones)

      if (!result || !result.variants) {
        if (!bypass) {
          logDebug('Rewrite failed, typing original text')
          typeText(text).catch(e => logDebug(`typeText failed: ${e.message}`))
        }
        return
      }

      const timeoutSec = this._settings.get_int('language-buddy-timeout')

      if (bypass) {
        this._overlay.show(result.variants, async (selectedText) => {
          if (selectedText === text) return
          try {
            await backspaceN(text.length)
            await typeText(selectedText)
          } catch (e) {
            logDebug(`Replace failed: ${e.message}`)
          }
        }, timeoutSec)
      } else {
        this._overlay.show(result.variants, async (selectedText) => {
          try {
            await typeText(selectedText)
          } catch (e) {
            logDebug(`typeText failed: ${e.message}`)
            Main.notify(_('Whisper NPU'), _('Failed to type selected text'))
          }
        }, timeoutSec)
      }
    }

    // -- Menu ---------------------------------------------------------------

    _buildMenu () {
      this._statusItem = new PopupMenu.PopupMenuItem(_('Status: Checking...'), { reactive: false })
      this._statusItem.label.add_style_class_name('whisper-status-label')
      this.menu.addMenuItem(this._statusItem)
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // Language Buddy toggle
      this._buddyToggle = new PopupMenu.PopupSwitchMenuItem(
        _('Language Buddy'),
        this._settings.get_boolean('language-buddy-enabled')
      )
      this._buddyToggle.connect('toggled', (_item, state) => {
        this._settings.set_boolean('language-buddy-enabled', state)
        logDebug(`Language Buddy ${state ? 'enabled' : 'disabled'}`)
      })
      this.menu.addMenuItem(this._buddyToggle)

      this._bypassToggle = new PopupMenu.PopupSwitchMenuItem(
        _('  Bypass (type original, show suggestions)'),
        this._settings.get_boolean('language-buddy-bypass')
      )
      this._bypassToggle.connect('toggled', (_item, state) => {
        this._settings.set_boolean('language-buddy-bypass', state)
        logDebug(`Language Buddy bypass ${state ? 'enabled' : 'disabled'}`)
      })
      this.menu.addMenuItem(this._bypassToggle)
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // STT model selection
      this._modelSection = new PopupMenu.PopupSubMenuMenuItem(_('Model'))
      this.menu.addMenuItem(this._modelSection)

      // STT model download
      this._downloadSection = new PopupMenu.PopupSubMenuMenuItem(_('Download Models'))
      this.menu.addMenuItem(this._downloadSection)
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // LLM model section
      this._llmModelSection = new PopupMenu.PopupSubMenuMenuItem(_('Language Buddy Models'))
      this.menu.addMenuItem(this._llmModelSection)
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // Server options
      this._deviceSection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Device: ${this._settings.get_string('device')}`)
      )
      this._buildRadioGroup(this._deviceSection, DEVICES, 'device')
      this.menu.addMenuItem(this._deviceSection)

      this._backendSection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Backend: ${this._settings.get_string('backend')}`)
      )
      this._buildRadioGroup(this._backendSection, BACKENDS, 'backend')
      this.menu.addMenuItem(this._backendSection)

      this._hotkeySection = new PopupMenu.PopupSubMenuMenuItem(
        _(`Hotkey: ${this._formatHotkey(this._settings.get_string('hotkey'))}`)
      )
      this._buildHotkeyGroup(this._hotkeySection)
      this.menu.addMenuItem(this._hotkeySection)

      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      const applyItem = new PopupMenu.PopupMenuItem(_('Apply & Restart Services'))
      applyItem.connect('activate', () => this._applyAndRestart())
      this.menu.addMenuItem(applyItem)

      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      const prefsItem = new PopupMenu.PopupMenuItem(_('Preferences...'))
      prefsItem.connect('activate', () => {
        this._extension.openPreferences()
      })
      this.menu.addMenuItem(prefsItem)
    }

    // -- Radio groups -------------------------------------------------------

    _buildRadioGroup (section, options, settingKey) {
      const current = this._settings.get_string(settingKey)
      for (const option of options) {
        const item = new PopupMenu.PopupMenuItem(option)
        if (option === current) {
          item.setOrnament(PopupMenu.Ornament.DOT)
        }
        item.connect('activate', () => {
          this._settings.set_string(settingKey, option)
          section.label.set_text(`${settingKey.charAt(0).toUpperCase() + settingKey.slice(1)}: ${option}`)
          this._updateRadioOrnaments(section, option)
        })
        section.menu.addMenuItem(item)
      }
    }

    _buildHotkeyGroup (section) {
      const current = this._settings.get_string('hotkey')
      for (const key of HOTKEYS) {
        const label = this._formatHotkey(key)
        const item = new PopupMenu.PopupMenuItem(label)
        if (key === current) {
          item.setOrnament(PopupMenu.Ornament.DOT)
        }
        item.connect('activate', () => {
          this._settings.set_string('hotkey', key)
          section.label.set_text(`Hotkey: ${label}`)
          this._updateHotkeyOrnaments(section, key)
        })
        section.menu.addMenuItem(item)
      }
    }

    _updateRadioOrnaments (section, selected) {
      const items = section.menu._getMenuItems()
      for (const item of items) {
        if (item.label) {
          item.setOrnament(
            item.label.get_text() === selected
              ? PopupMenu.Ornament.DOT
              : PopupMenu.Ornament.NONE
          )
        }
      }
    }

    _updateHotkeyOrnaments (section, selectedKey) {
      const items = section.menu._getMenuItems()
      for (const [i, item] of items.entries()) {
        if (item.label) {
          item.setOrnament(
            HOTKEYS[i] === selectedKey
              ? PopupMenu.Ornament.DOT
              : PopupMenu.Ornament.NONE
          )
        }
      }
    }

    _formatHotkey (key) {
      return key.replace('KEY_', '').replace(/([A-Z])([A-Z]+)/g, (_, first, rest) => {
        return first + rest.toLowerCase()
      })
    }

    // -- Server status ------------------------------------------------------

    async _refreshStatus () {
      this._statusItem.label.set_text(_('Status: Checking...'))

      const health = await this._client.getHealth()
      if (health) {
        this._connected = true
        this._currentModel = health.model
        this._statusItem.label.set_text(_(`Status: Connected | Model: ${health.model}`))
      } else {
        this._connected = false
        this._currentModel = null
        this._statusItem.label.set_text(_('Status: Disconnected'))
      }

      this._populateModelSection()
    }

    // -- STT model management -----------------------------------------------

    _populateModelSection () {
      this._modelSection.menu.removeAll()

      const localModels = listLocalModels()

      if (localModels.length === 0) {
        const noModels = new PopupMenu.PopupMenuItem(_('No models found'), { reactive: false })
        this._modelSection.menu.addMenuItem(noModels)
        return
      }

      for (const model of localModels) {
        const item = new PopupMenu.PopupMenuItem(model)
        if (model === this._currentModel) {
          item.setOrnament(PopupMenu.Ornament.DOT)
        }
        item.connect('activate', () => this._switchModel(model))
        this._modelSection.menu.addMenuItem(item)
      }

      const activeLabel = this._currentModel || localModels[0]
      this._modelSection.label.set_text(_(`Model: ${activeLabel}`))
    }

    async _switchModel (modelName) {
      if (!this._connected) {
        Main.notify(_('Whisper NPU'), _('Server is not connected'))
        return
      }

      logDebug(`Switching to model: ${modelName}`)
      const result = await this._client.setDefaultModel(modelName)

      if (result) {
        this._currentModel = modelName
        this._modelSection.label.set_text(_(`Model: ${modelName}`))
        this._refreshModelOrnaments(modelName)
        Main.notify(_('Whisper NPU'), _(`Switched to model: ${modelName}`))
      } else {
        Main.notify(_('Whisper NPU'), _('Failed to switch model'))
      }
    }

    _refreshModelOrnaments (selected) {
      const items = this._modelSection.menu._getMenuItems()
      for (const item of items) {
        if (item.label) {
          item.setOrnament(
            item.label.get_text() === selected
              ? PopupMenu.Ornament.DOT
              : PopupMenu.Ornament.NONE
          )
        }
      }
    }

    // -- STT model download -------------------------------------------------

    _populateDownloadSection () {
      this._downloadSection.menu.removeAll()

      const org = this._settings.get_string('hf-org')
      const remoteModels = this._cachedRemoteModels
      const localModels = listLocalModels()

      if (!this._cacheReady) {
        const loadingItem = new PopupMenu.PopupMenuItem(_('Loading...'), { reactive: false })
        this._downloadSection.menu.addMenuItem(loadingItem)
        return
      }

      if (remoteModels.length === 0) {
        const noModels = new PopupMenu.PopupMenuItem(_('No models found'), { reactive: false })
        this._downloadSection.menu.addMenuItem(noModels)
        return
      }

      const available = remoteModels.filter(m => !localModels.includes(m.name))

      if (available.length === 0) {
        const allInstalled = new PopupMenu.PopupMenuItem(_('All models installed'), { reactive: false })
        this._downloadSection.menu.addMenuItem(allInstalled)
      } else {
        for (const model of available) {
          const downloads = model.downloads > 1000
            ? `${(model.downloads / 1000).toFixed(0)}k`
            : `${model.downloads}`
          const item = new PopupMenu.PopupMenuItem(`${model.name}  (${downloads} downloads)`)
          item.connect('activate', () => this._startDownload(org, model.name))
          this._downloadSection.menu.addMenuItem(item)
        }
      }

      this._downloadSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      const refreshItem = new PopupMenu.PopupMenuItem(_('Refresh List'))
      refreshItem.connect('activate', () => this._refreshDownloadCache())
      this._downloadSection.menu.addMenuItem(refreshItem)
    }

    async _refreshDownloadCache () {
      const org = this._settings.get_string('hf-org')
      this._cachedRemoteModels = await this._hfClient.searchModels(org) || []
      this._populateDownloadSection()
    }

    async _startDownload (org, modelName) {
      Main.notify(_('Whisper NPU'), _(`Downloading ${modelName}...`))
      logDebug(`Starting download: ${org}/${modelName}`)

      try {
        await downloadModel(org, modelName)
        Main.notify(_('Whisper NPU'), _(`Downloaded ${modelName} successfully`))
        this._populateModelSection()
        this._populateDownloadSection()
      } catch (e) {
        logDebug(`Download failed: ${e.message}`)
        Main.notify(_('Whisper NPU'), _(`Download failed: ${e.message}`))
      }
    }

    // -- LLM model management -----------------------------------------------

    _populateLlmModelSection () {
      this._llmModelSection.menu.removeAll()

      const localModels = listLocalLlmModels()

      if (localModels.length === 0) {
        const noModels = new PopupMenu.PopupMenuItem(_('No LLM models found'), { reactive: false })
        this._llmModelSection.menu.addMenuItem(noModels)
      } else {
        for (const model of localModels) {
          const item = new PopupMenu.PopupMenuItem(model)
          if (model === this._currentLlmModel) {
            item.setOrnament(PopupMenu.Ornament.DOT)
          }
          item.connect('activate', () => this._switchLlmModel(model))
          this._llmModelSection.menu.addMenuItem(item)
        }
      }

      this._llmModelSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      const downloadItem = new PopupMenu.PopupMenuItem(_('Download LLM Model...'))
      downloadItem.connect('activate', () => this._startLlmDownload())
      this._llmModelSection.menu.addMenuItem(downloadItem)

      const label = this._currentLlmModel ? `LLM: ${this._currentLlmModel}` : 'Language Buddy Models'
      this._llmModelSection.label.set_text(_(label))
    }

    async _switchLlmModel (modelName) {
      logDebug(`Switching LLM to: ${modelName}`)
      const result = await this._client.setLlmModel(modelName)
      if (result) {
        Main.notify(_('Whisper NPU'), _(`LLM switched to: ${modelName}`))
        this._populateLlmModelSection()
      } else {
        Main.notify(_('Whisper NPU'), _('Failed to switch LLM model'))
      }
    }

    async _startLlmDownload () {
      const org = this._settings.get_string('hf-org')
      const modelName = 'Qwen2.5-1.5B-Instruct-int4-ov'
      Main.notify(_('Whisper NPU'), _(`Downloading ${modelName}... This may take a while.`))
      try {
        await downloadLlmModel(org, modelName)
        Main.notify(_('Whisper NPU'), _(`Downloaded ${modelName} successfully`))
        this._populateLlmModelSection()
      } catch (e) {
        logDebug(`LLM download failed: ${e.message}`)
        Main.notify(_('Whisper NPU'), _(`Download failed: ${e.message}`))
      }
    }

    // -- Apply settings -----------------------------------------------------

    async _applyAndRestart () {
      const device = this._settings.get_string('device')
      const backend = this._settings.get_string('backend')
      const hotkey = this._settings.get_string('hotkey')

      logDebug(`Applying settings: device=${device} backend=${backend} hotkey=${hotkey}`)

      try {
        await writeServiceOverride('whisper-server.service', {
          WHISPER_DEVICE: device
        })

        await restartService('whisper-server.service')
        await restartService('push-to-talk.service')

        Main.notify(_('Whisper NPU'), _('Services restarted with new settings'))
      } catch (e) {
        logDebug(`Failed to apply settings: ${e.message}`)
        Main.notify(_('Whisper NPU'), _(`Failed to restart: ${e.message}`))
      }
    }

    // -- Cleanup ------------------------------------------------------------

    destroy () {
      if (this._dbusImpl) {
        this._dbusImpl.unexport()
        this._dbusImpl = null
      }
      if (this._overlay) {
        Main.layoutManager.removeChrome(this._overlay)
        this._overlay.destroy()
        this._overlay = null
      }
      if (this._settingsChangedId) {
        this._settings.disconnect(this._settingsChangedId)
        this._settingsChangedId = null
      }
      if (this._downloadCancellable) {
        this._downloadCancellable.cancel()
        this._downloadCancellable = null
      }
      if (this._client) {
        this._client.destroy()
        this._client = null
      }
      if (this._hfClient) {
        this._hfClient.destroy()
        this._hfClient = null
      }
      this._settings = null
      super.destroy()
    }
  }
)

// ---------------------------------------------------------------------------
// Extension lifecycle
// ---------------------------------------------------------------------------

export default class WhisperNpuExtension extends Extension {
  enable () {
    this._settings = this.getSettings()
    this._indicator = new WhisperIndicator(this)
    Main.panel.addToStatusArea('whisper-npu', this._indicator)
  }

  disable () {
    if (this._indicator) {
      this._indicator.destroy()
      this._indicator = null
      this._settings = null
    }
  }
}

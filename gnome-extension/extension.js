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

    show (tones, originalText, onSelect, timeoutSec) {
      this.destroy_all_children()
      this._cards = {}
      this._onSelect = onSelect

      const headerBox = new St.BoxLayout({ x_expand: true })
      const headerLabel = new St.Label({
        text: _('Language Buddy'),
        style_class: 'whisper-overlay-header',
        x_expand: true
      })
      const dismissBtn = new St.Button({
        style_class: 'whisper-overlay-dismiss',
        child: new St.Label({ text: '✕' })
      })
      dismissBtn.connect('clicked', () => this._dismiss())
      headerBox.add_child(headerLabel)
      headerBox.add_child(dismissBtn)
      this.add_child(headerBox)

      this._addCard('original', originalText, true)
      for (const tone of tones) {
        this._addCard(tone, _('Processing...'), false)
      }

      this._showAndPosition()

      if (timeoutSec > 0) {
        this._timeoutId = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, timeoutSec, () => {
          this._dismiss()
          return GLib.SOURCE_REMOVE
        })
      }
    }

    _addCard (tone, text, ready) {
      const card = new St.Button({
        style_class: 'whisper-overlay-card',
        x_expand: true,
        reactive: ready,
        can_focus: false,
        track_hover: true
      })

      const cardBox = new St.BoxLayout({ vertical: true })

      const toneLabel = new St.Label({
        text: tone.charAt(0).toUpperCase() + tone.slice(1),
        style_class: 'whisper-overlay-tone'
      })
      cardBox.add_child(toneLabel)

      const textLabel = new St.Label({
        text,
        style_class: 'whisper-overlay-text'
      })
      textLabel.clutter_text.set_line_wrap(true)
      textLabel.clutter_text.set_ellipsize(0)
      cardBox.add_child(textLabel)

      card.set_child(cardBox)

      const entry = { card, textLabel, text }
      this._cards[tone] = entry

      card.connect('clicked', () => {
        if (this._onSelect && entry.card.reactive) {
          const selectedText = entry.text
          const callback = this._onSelect
          this._dismiss()
          callback(selectedText)
        }
      })

      this.add_child(card)
    }

    updateCard (tone, text) {
      const entry = this._cards[tone]
      if (!entry) return
      entry.text = text
      entry.textLabel.set_text(text)
      entry.card.reactive = true
      this._reposition()
    }

    _showAndPosition () {
      const monitor = Main.layoutManager.primaryMonitor
      if (!monitor) return

      const maxH = monitor.height - OVERLAY_PADDING * 2
      this.style = `max-height: ${maxH}px;`

      this.set_position(monitor.x + monitor.width, monitor.y + monitor.height)
      this.visible = true

      this._reposition()
    }

    _reposition () {
      if (this._positionId) {
        GLib.source_remove(this._positionId)
        this._positionId = null
      }
      this._positionId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 50, () => {
        this._positionId = null
        const monitor = Main.layoutManager.primaryMonitor
        if (!monitor) return GLib.SOURCE_REMOVE
        this.set_position(
          monitor.x + monitor.width - this.width - OVERLAY_PADDING,
          monitor.y + monitor.height - this.height - OVERLAY_PADDING
        )
        return GLib.SOURCE_REMOVE
      })
    }

    _dismiss () {
      if (this._positionId) {
        GLib.source_remove(this._positionId)
        this._positionId = null
      }
      if (this._timeoutId) {
        GLib.source_remove(this._timeoutId)
        this._timeoutId = null
      }
      this.visible = false
      this.style = null
      this._cards = {}
      this._onSelect = null
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
      this._cachedRemoteModels = []
      this._cachedRemoteLlmModels = []
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
        const [health, remoteModels, remoteLlmModels, llmInfo] = await Promise.all([
          this._client.getHealth(),
          this._hfClient.searchModels(org),
          this._hfClient.searchLlmModels(org),
          this._client.getLlmModels()
        ])

        if (health) {
          this._connected = true
          this._currentModel = health.model
        }

        this._currentLlmModel = llmInfo ? llmInfo.current : null

        this._cachedRemoteModels = remoteModels || []
        this._cachedRemoteLlmModels = remoteLlmModels || []
        this._cacheReady = true
        logDebug(`Preloaded ${this._cachedRemoteModels.length} STT and ${this._cachedRemoteLlmModels.length} LLM remote models`)

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

      const timeoutSec = this._settings.get_int('language-buddy-timeout')
      const tones = DEFAULT_TONES

      const onSelect = bypass
        ? async (selectedText) => {
            if (selectedText === text) return
            try {
              await backspaceN(text.length)
              await typeText(selectedText)
            } catch (e) {
              logDebug(`Replace failed: ${e.message}`)
            }
          }
        : async (selectedText) => {
            try {
              await typeText(selectedText)
            } catch (e) {
              logDebug(`typeText failed: ${e.message}`)
            }
          }

      this._overlay.show(tones, text, onSelect, timeoutSec)

      for (const tone of tones) {
        try {
          const result = await this._client.rewrite(text, [tone])
          if (result && result.variants) {
            const variant = result.variants.find(v => v.tone === tone)
            if (variant && !variant.error) {
              this._overlay.updateCard(tone, variant.text)
            } else {
              this._overlay.updateCard(tone, text)
            }
          } else {
            this._overlay.updateCard(tone, text)
          }
        } catch (e) {
          logDebug(`Rewrite (${tone}) failed: ${e.message}`)
          this._overlay.updateCard(tone, text)
        }
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

      // STT models (installed + available downloads)
      this._modelSection = new PopupMenu.PopupSubMenuMenuItem(_('Speech-to-Text Models'))
      this._modelSection.menu.addMenuItem(
        new PopupMenu.PopupMenuItem(_('Loading...'), { reactive: false })
      )
      this.menu.addMenuItem(this._modelSection)
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      // LLM model section
      this._llmModelSection = new PopupMenu.PopupSubMenuMenuItem(_('Language Buddy Models'))
      this._llmModelSection.menu.addMenuItem(
        new PopupMenu.PopupMenuItem(_('Loading...'), { reactive: false })
      )
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

      this._applyItem = new PopupMenu.PopupMenuItem(_('Apply & Restart Services'))
      this._applyItem.connect('activate', () => this._applyAndRestart())
      this.menu.addMenuItem(this._applyItem)

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
        const noModels = new PopupMenu.PopupMenuItem(_('No models installed'), { reactive: false })
        this._modelSection.menu.addMenuItem(noModels)
      } else {
        for (const model of localModels) {
          const item = new PopupMenu.PopupMenuItem(model)
          if (model === this._currentModel) {
            item.setOrnament(PopupMenu.Ornament.DOT)
          }
          item.connect('activate', () => this._switchModel(model, item))
          this._modelSection.menu.addMenuItem(item)
        }
      }

      const available = this._cachedRemoteModels.filter(
        m => !localModels.includes(m.name)
      )

      if (available.length > 0) {
        this._modelSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

        const downloadHeader = new PopupMenu.PopupMenuItem(_('Available Downloads'), { reactive: false })
        downloadHeader.label.add_style_class_name('whisper-status-label')
        this._modelSection.menu.addMenuItem(downloadHeader)

        const org = this._settings.get_string('hf-org')
        for (const model of available) {
          const downloads = model.downloads > 1000
            ? `${(model.downloads / 1000).toFixed(0)}k`
            : `${model.downloads}`
          const item = new PopupMenu.PopupMenuItem(`${model.name}  (${downloads} downloads)`)
          item.connect('activate', () => this._startDownload(org, model.name, item))
          this._modelSection.menu.addMenuItem(item)
        }
      }

      this._modelSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      const refreshItem = new PopupMenu.PopupMenuItem(_('Refresh List'))
      refreshItem.connect('activate', () => {
        refreshItem.label.set_text(_('Refreshing...'))
        refreshItem.reactive = false
        this._refreshModelCache()
      })
      this._modelSection.menu.addMenuItem(refreshItem)

      const activeLabel = this._currentModel || (localModels.length > 0 ? localModels[0] : '')
      if (activeLabel) {
        this._modelSection.label.set_text(_(`STT: ${activeLabel}`))
      }
    }

    async _switchModel (modelName, item) {
      if (!this._connected) {
        Main.notify(_('Whisper NPU'), _('Server is not connected'))
        return
      }

      const origLabel = item.label.get_text()
      item.label.set_text(_(`Loading ${modelName}...`))
      item.reactive = false
      this._modelSection.label.set_text(_(`Loading ${modelName}...`))

      logDebug(`Switching to model: ${modelName}`)
      const result = await this._client.setDefaultModel(modelName)

      if (result) {
        this._currentModel = modelName
        Main.notify(_('Whisper NPU'), _(`Switched to ${modelName}`))
      } else {
        item.label.set_text(origLabel)
        item.reactive = true
        Main.notify(_('Whisper NPU'), _(`Failed to load ${modelName}`))
      }
      this._populateModelSection()
    }

    // -- STT model download -------------------------------------------------

    async _refreshModelCache () {
      const org = this._settings.get_string('hf-org')
      this._cachedRemoteModels = await this._hfClient.searchModels(org) || []
      this._populateModelSection()
    }

    async _startDownload (org, modelName, item) {
      item.label.set_text(_(`Downloading ${modelName}...`))
      item.reactive = false
      this._modelSection.label.set_text(_(`Downloading ${modelName}...`))
      logDebug(`Starting download: ${org}/${modelName}`)

      try {
        await downloadModel(org, modelName)
        item.label.set_text(_(`Downloaded ${modelName}`))
        Main.notify(_('Whisper NPU'), _(`Downloaded ${modelName}`))
      } catch (e) {
        logDebug(`Download failed: ${e.message}`)
        item.label.set_text(_(`Failed: ${modelName}`))
        Main.notify(_('Whisper NPU'), _(`Download failed: ${e.message}`))
      }
      this._populateModelSection()
    }

    // -- LLM model management -----------------------------------------------

    _populateLlmModelSection () {
      this._llmModelSection.menu.removeAll()

      const localModels = listLocalLlmModels()

      if (localModels.length === 0) {
        const noModels = new PopupMenu.PopupMenuItem(_('No LLM models installed'), { reactive: false })
        this._llmModelSection.menu.addMenuItem(noModels)
      } else {
        for (const model of localModels) {
          const item = new PopupMenu.PopupMenuItem(model)
          if (model === this._currentLlmModel) {
            item.setOrnament(PopupMenu.Ornament.DOT)
          }
          item.connect('activate', () => this._switchLlmModel(model, item))
          this._llmModelSection.menu.addMenuItem(item)
        }
      }

      const available = this._cachedRemoteLlmModels.filter(
        m => !localModels.includes(m.name)
      )

      if (available.length > 0) {
        this._llmModelSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

        const downloadHeader = new PopupMenu.PopupMenuItem(_('Available Downloads'), { reactive: false })
        downloadHeader.label.add_style_class_name('whisper-status-label')
        this._llmModelSection.menu.addMenuItem(downloadHeader)

        for (const model of available) {
          const downloads = model.downloads > 1000
            ? `${(model.downloads / 1000).toFixed(0)}k`
            : `${model.downloads}`
          const item = new PopupMenu.PopupMenuItem(`${model.name}  (${downloads} downloads)`)
          item.connect('activate', () => this._startLlmDownload(model.name, item))
          this._llmModelSection.menu.addMenuItem(item)
        }
      }

      this._llmModelSection.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem())

      const refreshItem = new PopupMenu.PopupMenuItem(_('Refresh List'))
      refreshItem.connect('activate', () => {
        refreshItem.label.set_text(_('Refreshing...'))
        refreshItem.reactive = false
        this._refreshLlmCache()
      })
      this._llmModelSection.menu.addMenuItem(refreshItem)

      const label = this._currentLlmModel ? `LLM: ${this._currentLlmModel}` : 'Language Buddy Models'
      this._llmModelSection.label.set_text(_(label))
    }

    async _refreshLlmCache () {
      const org = this._settings.get_string('hf-org')
      this._cachedRemoteLlmModels = await this._hfClient.searchLlmModels(org) || []
      this._populateLlmModelSection()
    }

    async _switchLlmModel (modelName, item) {
      const origLabel = item.label.get_text()
      item.label.set_text(_(`Loading ${modelName}...`))
      item.reactive = false
      this._llmModelSection.label.set_text(_(`Loading ${modelName}...`))

      logDebug(`Switching LLM to: ${modelName}`)
      const result = await this._client.setLlmModel(modelName)

      if (result) {
        this._currentLlmModel = modelName
        Main.notify(_('Whisper NPU'), _(`LLM: ${modelName}`))
      } else {
        item.label.set_text(origLabel)
        item.reactive = true
        Main.notify(_('Whisper NPU'), _(`Failed to load ${modelName}`))
      }
      this._populateLlmModelSection()
    }

    async _startLlmDownload (modelName, item) {
      logDebug(`Download LLM: ${modelName}`)
      item.label.set_text(_(`Downloading ${modelName}...`))
      item.reactive = false
      this._llmModelSection.label.set_text(_(`Downloading ${modelName}...`))
      const org = this._settings.get_string('hf-org')

      try {
        await downloadLlmModel(org, modelName)
        item.label.set_text(_(`Downloaded ${modelName}`))
        Main.notify(_('Whisper NPU'), _(`Downloaded ${modelName}`))
      } catch (e) {
        logDebug(`LLM download failed: ${e.message}`)
        item.label.set_text(_(`Failed: ${modelName}`))
        Main.notify(_('Whisper NPU'), _(`Download failed: ${e.message}`))
      }
      this._populateLlmModelSection()
    }

    // -- Apply settings -----------------------------------------------------

    async _applyAndRestart () {
      this._applyItem.label.set_text(_('Restarting services...'))
      this._applyItem.reactive = false

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

        this._applyItem.label.set_text(_('Services restarted'))
        Main.notify(_('Whisper NPU'), _('Services restarted'))
      } catch (e) {
        logDebug(`Failed to apply settings: ${e.message}`)
        this._applyItem.label.set_text(_('Restart failed'))
        Main.notify(_('Whisper NPU'), _(`Failed to restart: ${e.message}`))
      }

      this._applyItem.reactive = true
      GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, 3, () => {
        this._applyItem.label.set_text(_('Apply & Restart Services'))
        return GLib.SOURCE_REMOVE
      })
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

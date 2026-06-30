import Adw from 'gi://Adw'
import Gtk from 'gi://Gtk'
import Gio from 'gi://Gio'

import { ExtensionPreferences } from 'resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js'

export default class WhisperNpuPreferences extends ExtensionPreferences {
  fillPreferencesWindow (window) {
    const settings = this.getSettings()

    const page = new Adw.PreferencesPage()

    page.add(this._buildServerGroup(settings))
    page.add(this._buildOptionsGroup(settings))

    window.add(page)
  }

  _buildServerGroup (settings) {
    const group = new Adw.PreferencesGroup({
      title: 'Server Connection'
    })

    const hostEntry = new Gtk.Entry({ hexpand: true })
    hostEntry.set_text(settings.get_string('server-host'))
    hostEntry.connect('changed', () => {
      settings.set_string('server-host', hostEntry.get_text())
    })

    const hostRow = new Adw.ActionRow({ title: 'Host' })
    hostRow.add_suffix(hostEntry)
    hostRow.activatable = false
    group.add(hostRow)

    const portAdj = new Gtk.Adjustment({
      lower: 1024,
      upper: 65535,
      step_increment: 1,
      value: settings.get_int('server-port')
    })
    const portSpin = new Gtk.SpinButton({
      adjustment: portAdj,
      numeric: true,
      valign: Gtk.Align.CENTER
    })
    portSpin.connect('value-changed', () => {
      settings.set_int('server-port', portSpin.get_value_as_int())
    })

    const portRow = new Adw.ActionRow({ title: 'Port' })
    portRow.add_suffix(portSpin)
    portRow.activatable = false
    group.add(portRow)

    return group
  }

  _buildOptionsGroup (settings) {
    const group = new Adw.PreferencesGroup({
      title: 'Server Options'
    })

    const deviceCombo = new Gtk.ComboBoxText({ valign: Gtk.Align.CENTER })
    for (const d of ['NPU', 'CPU', 'GPU']) { deviceCombo.append_text(d) }
    deviceCombo.set_active(['NPU', 'CPU', 'GPU'].indexOf(settings.get_string('device')))
    deviceCombo.connect('changed', () => {
      settings.set_string('device', deviceCombo.get_active_text())
    })

    const deviceRow = new Adw.ActionRow({ title: 'Device' })
    deviceRow.add_suffix(deviceCombo)
    deviceRow.activatable = false
    group.add(deviceRow)

    const backendCombo = new Gtk.ComboBoxText({ valign: Gtk.Align.CENTER })
    for (const b of ['openvino', 'whisper-cpp']) { backendCombo.append_text(b) }
    backendCombo.set_active(['openvino', 'whisper-cpp'].indexOf(settings.get_string('backend')))
    backendCombo.connect('changed', () => {
      settings.set_string('backend', backendCombo.get_active_text())
    })

    const backendRow = new Adw.ActionRow({ title: 'Backend' })
    backendRow.add_suffix(backendCombo)
    backendRow.activatable = false
    group.add(backendRow)

    const hotkeyEntry = new Gtk.Entry({ hexpand: true })
    hotkeyEntry.set_text(settings.get_string('hotkey'))
    hotkeyEntry.connect('changed', () => {
      settings.set_string('hotkey', hotkeyEntry.get_text())
    })

    const hotkeyRow = new Adw.ActionRow({ title: 'Hotkey (evdev key name)' })
    hotkeyRow.add_suffix(hotkeyEntry)
    hotkeyRow.activatable = false
    group.add(hotkeyRow)

    const intervalAdj = new Gtk.Adjustment({
      lower: 0.5,
      upper: 30.0,
      step_increment: 0.5,
      value: settings.get_double('stream-interval')
    })
    const intervalSpin = new Gtk.SpinButton({
      adjustment: intervalAdj,
      digits: 1,
      numeric: true,
      valign: Gtk.Align.CENTER
    })
    intervalSpin.connect('value-changed', () => {
      settings.set_double('stream-interval', intervalSpin.get_value())
    })

    const intervalRow = new Adw.ActionRow({ title: 'Stream Interval (seconds)' })
    intervalRow.add_suffix(intervalSpin)
    intervalRow.activatable = false
    group.add(intervalRow)

    const hfEntry = new Gtk.Entry({ hexpand: true })
    hfEntry.set_text(settings.get_string('hf-org'))
    hfEntry.connect('changed', () => {
      settings.set_string('hf-org', hfEntry.get_text())
    })

    const hfRow = new Adw.ActionRow({ title: 'HuggingFace Organization' })
    hfRow.add_suffix(hfEntry)
    hfRow.activatable = false
    group.add(hfRow)

    return group
  }
}

import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.plasma.components as PlasmaComponents
import org.kde.plasma.extras as PlasmaExtras
import org.kde.kirigami as Kirigami

PlasmaExtras.Representation {
    id: fullRep

    property string serverHost
    property int serverPort
    property bool connected
    property string currentModel
    property string currentLlmModel
    property var localModels: []
    property var localLlmModels: []
    property var remoteModels: []
    property var remoteLlmModels: []

    property var devices: ["NPU", "CPU", "GPU"]
    property var backends: ["openvino", "whisper-cpp"]
    property var hotkeys: ["KEY_RIGHTCTRL", "KEY_RIGHTALT", "KEY_RIGHTSHIFT", "KEY_SCROLLLOCK", "KEY_PAUSE"]
    property var languages: [
        {label: "Auto", code: ""},
        {label: "English", code: "en"},
        {label: "German", code: "de"},
        {label: "French", code: "fr"},
        {label: "Spanish", code: "es"},
        {label: "Italian", code: "it"},
        {label: "Portuguese", code: "pt"},
        {label: "Dutch", code: "nl"},
        {label: "Polish", code: "pl"},
        {label: "Japanese", code: "ja"},
        {label: "Chinese", code: "zh"},
        {label: "Korean", code: "ko"}
    ]
    property var translateTargets: [
        {label: "Disabled", value: ""},
        {label: "English", value: "English"},
        {label: "German", value: "German"},
        {label: "French", value: "French"},
        {label: "Spanish", value: "Spanish"},
        {label: "Italian", value: "Italian"},
        {label: "Portuguese", value: "Portuguese"},
        {label: "Dutch", value: "Dutch"},
        {label: "Polish", value: "Polish"},
        {label: "Japanese", value: "Japanese"},
        {label: "Chinese", value: "Chinese"},
        {label: "Korean", value: "Korean"}
    ]
    property var vadOptions: [
        {label: "-20 dB (aggressive)", value: -20},
        {label: "-30 dB", value: -30},
        {label: "-40 dB (default)", value: -40},
        {label: "-50 dB", value: -50},
        {label: "-60 dB (sensitive)", value: -60}
    ]
    property var intervalOptions: [
        {label: "1.0s (fast)", value: 1.0},
        {label: "2.0s", value: 2.0},
        {label: "3.0s (default)", value: 3.0},
        {label: "5.0s", value: 5.0},
        {label: "10.0s (slow)", value: 10.0}
    ]

    implicitWidth: Kirigami.Units.gridUnit * 22
    implicitHeight: Kirigami.Units.gridUnit * 35

    function formatHotkey(key) {
        return key.replace("KEY_", "").replace(/([A-Z])([A-Z]+)/g, function(_, first, rest) {
            return first + rest.toLowerCase()
        })
    }

    function formatDownloads(n) {
        return n > 1000 ? Math.floor(n / 1000) + "k" : String(n)
    }

    function findIndex(arr, prop, value) {
        for (var i = 0; i < arr.length; i++) {
            if (arr[i][prop] === value) return i
        }
        return 0
    }

    header: PlasmaExtras.PlasmoidHeading {
        RowLayout {
            anchors.fill: parent

            PlasmaExtras.Heading {
                Layout.fillWidth: true
                level: 3
                text: "Whisper NPU"
            }

            PlasmaComponents.Label {
                text: connected ? "Connected" : "Disconnected"
                color: connected ? Kirigami.Theme.positiveTextColor : Kirigami.Theme.negativeTextColor
                font.italic: true
            }
        }
    }

    PlasmaComponents.ScrollView {
        anchors.fill: parent

        contentItem: Flickable {
            contentHeight: mainColumn.height

            ColumnLayout {
                id: mainColumn
                width: parent.width
                spacing: Kirigami.Units.smallSpacing

                // Status
                PlasmaComponents.Label {
                    Layout.fillWidth: true
                    text: connected ? "Model: " + currentModel : "Server not responding"
                    font.italic: true
                    opacity: 0.7
                    visible: currentModel || !connected
                }

                // ----- Language Buddy -----
                Kirigami.Separator { Layout.fillWidth: true }

                SwitchRow {
                    label: "Language Buddy"
                    checked: Plasmoid.configuration.languageBuddyEnabled
                    onToggled: function(state) {
                        Plasmoid.configuration.languageBuddyEnabled = state
                        root.saveSettings()
                    }
                }

                SwitchRow {
                    label: "  Bypass (type original, show suggestions)"
                    checked: Plasmoid.configuration.languageBuddyBypass
                    onToggled: function(state) {
                        Plasmoid.configuration.languageBuddyBypass = state
                        root.saveSettings()
                    }
                }

                Kirigami.Separator { Layout.fillWidth: true }

                // ----- STT Models -----
                CollapsibleSection {
                    title: currentModel ? "STT: " + currentModel : "Speech-to-Text Models"
                    Layout.fillWidth: true

                    content: ColumnLayout {
                        spacing: Kirigami.Units.smallSpacing
                        width: parent.width

                        Repeater {
                            model: localModels

                            PlasmaComponents.ItemDelegate {
                                Layout.fillWidth: true
                                text: modelData
                                icon.name: modelData === currentModel ? "emblem-default" : ""
                                highlighted: modelData === currentModel

                                onClicked: {
                                    if (modelData !== currentModel) {
                                        root.httpRequest("PUT", root.serverUrl("/model/default"),
                                            {model: modelData}, function(data, err) {
                                                if (data) {
                                                    root.currentModel = modelData
                                                    root.refreshStatus()
                                                }
                                            })
                                    }
                                }
                            }
                        }

                        PlasmaComponents.Label {
                            text: "No models installed"
                            visible: localModels.length === 0
                            font.italic: true
                            opacity: 0.6
                        }

                        Kirigami.Separator {
                            Layout.fillWidth: true
                            visible: availableSttRepeater.count > 0
                        }

                        PlasmaComponents.Label {
                            text: "Available Downloads"
                            font.italic: true
                            opacity: 0.6
                            visible: availableSttRepeater.count > 0
                        }

                        Repeater {
                            id: availableSttRepeater
                            model: {
                                var available = []
                                for (var i = 0; i < remoteModels.length; i++) {
                                    var found = false
                                    for (var j = 0; j < localModels.length; j++) {
                                        if (localModels[j] === remoteModels[i].name) {
                                            found = true
                                            break
                                        }
                                    }
                                    if (!found) available.push(remoteModels[i])
                                }
                                return available
                            }

                            PlasmaComponents.ItemDelegate {
                                Layout.fillWidth: true
                                text: modelData.name + "  (" + formatDownloads(modelData.downloads) + " downloads)"

                                property bool downloading: false

                                onClicked: {
                                    if (downloading) return
                                    downloading = true
                                    text = "Downloading " + modelData.name + "..."
                                    var org = Plasmoid.configuration.hfOrg || "OpenVINO"
                                    var dest = root.homeDir + "/.whisper/models/" + modelData.name
                                    var url = "https://huggingface.co/" + org + "/" + modelData.name
                                    root.executable.exec("git clone " + url + " " + dest + " && git -C " + dest + " lfs pull")
                                    root.refreshLocalModels()
                                }
                            }
                        }

                        PlasmaComponents.Button {
                            text: "Refresh List"
                            icon.name: "view-refresh"
                            onClicked: {
                                root.refreshRemoteModels()
                                root.refreshLocalModels()
                            }
                        }
                    }
                }

                Kirigami.Separator { Layout.fillWidth: true }

                // ----- LLM Models -----
                CollapsibleSection {
                    title: currentLlmModel ? "LLM: " + currentLlmModel : "Language Buddy Models"
                    Layout.fillWidth: true

                    content: ColumnLayout {
                        spacing: Kirigami.Units.smallSpacing
                        width: parent.width

                        Repeater {
                            model: localLlmModels

                            PlasmaComponents.ItemDelegate {
                                Layout.fillWidth: true
                                text: modelData
                                icon.name: modelData === currentLlmModel ? "emblem-default" : ""
                                highlighted: modelData === currentLlmModel

                                onClicked: {
                                    if (modelData !== currentLlmModel) {
                                        root.httpRequest("PUT", root.serverUrl("/llm/model"),
                                            {model: modelData}, function(data, err) {
                                                if (data) {
                                                    root.currentLlmModel = modelData
                                                }
                                            })
                                    }
                                }
                            }
                        }

                        PlasmaComponents.Label {
                            text: "No LLM models installed"
                            visible: localLlmModels.length === 0
                            font.italic: true
                            opacity: 0.6
                        }

                        Kirigami.Separator {
                            Layout.fillWidth: true
                            visible: availableLlmRepeater.count > 0
                        }

                        PlasmaComponents.Label {
                            text: "Available Downloads"
                            font.italic: true
                            opacity: 0.6
                            visible: availableLlmRepeater.count > 0
                        }

                        Repeater {
                            id: availableLlmRepeater
                            model: {
                                var available = []
                                for (var i = 0; i < remoteLlmModels.length; i++) {
                                    var found = false
                                    for (var j = 0; j < localLlmModels.length; j++) {
                                        if (localLlmModels[j] === remoteLlmModels[i].name) {
                                            found = true
                                            break
                                        }
                                    }
                                    if (!found) available.push(remoteLlmModels[i])
                                }
                                return available
                            }

                            PlasmaComponents.ItemDelegate {
                                Layout.fillWidth: true
                                text: modelData.name + "  (" + formatDownloads(modelData.downloads) + " downloads)"

                                property bool downloading: false

                                onClicked: {
                                    if (downloading) return
                                    downloading = true
                                    text = "Downloading " + modelData.name + "..."
                                    var org = Plasmoid.configuration.hfOrg || "OpenVINO"
                                    var dest = root.homeDir + "/.whisper/llm-models/" + modelData.name
                                    var url = "https://huggingface.co/" + org + "/" + modelData.name
                                    root.executable.exec("mkdir -p " + root.homeDir + "/.whisper/llm-models && git clone " + url + " " + dest + " && git -C " + dest + " lfs pull")
                                    root.refreshLocalModels()
                                }
                            }
                        }

                        PlasmaComponents.Button {
                            text: "Refresh List"
                            icon.name: "view-refresh"
                            onClicked: {
                                root.refreshRemoteModels()
                                root.refreshLocalModels()
                            }
                        }
                    }
                }

                Kirigami.Separator { Layout.fillWidth: true }

                // ----- Server Options -----
                ComboRow {
                    label: "Device"
                    model: devices
                    currentValue: Plasmoid.configuration.device || "NPU"
                    onSelected: function(value) {
                        Plasmoid.configuration.device = value
                        root.saveSettings()
                    }
                }

                ComboRow {
                    label: "Backend"
                    model: backends
                    currentValue: Plasmoid.configuration.backend || "openvino"
                    onSelected: function(value) {
                        Plasmoid.configuration.backend = value
                        root.saveSettings()
                    }
                }

                ComboRow {
                    label: "Hotkey"
                    model: hotkeys.map(formatHotkey)
                    currentValue: formatHotkey(Plasmoid.configuration.hotkey || "KEY_RIGHTCTRL")
                    onSelected: function(value) {
                        for (var i = 0; i < hotkeys.length; i++) {
                            if (formatHotkey(hotkeys[i]) === value) {
                                Plasmoid.configuration.hotkey = hotkeys[i]
                                root.saveSettings()
                                break
                            }
                        }
                    }
                }

                ComboRow {
                    label: "Language"
                    model: languages.map(function(l) { return l.label })
                    currentValue: {
                        var code = Plasmoid.configuration.language || ""
                        for (var i = 0; i < languages.length; i++) {
                            if (languages[i].code === code) return languages[i].label
                        }
                        return "Auto"
                    }
                    onSelected: function(value) {
                        for (var i = 0; i < languages.length; i++) {
                            if (languages[i].label === value) {
                                Plasmoid.configuration.language = languages[i].code
                                root.saveSettings()
                                break
                            }
                        }
                    }
                }

                ComboRow {
                    label: "Recall Key"
                    model: hotkeys.map(formatHotkey)
                    currentValue: formatHotkey(Plasmoid.configuration.recallKey || "KEY_PAUSE")
                    onSelected: function(value) {
                        for (var i = 0; i < hotkeys.length; i++) {
                            if (formatHotkey(hotkeys[i]) === value) {
                                Plasmoid.configuration.recallKey = hotkeys[i]
                                root.saveSettings()
                                break
                            }
                        }
                    }
                }

                Kirigami.Separator { Layout.fillWidth: true }

                // ----- Feature Toggles -----
                SwitchRow {
                    label: "Voice Commands"
                    checked: Plasmoid.configuration.voiceCommandsEnabled
                    onToggled: function(state) {
                        Plasmoid.configuration.voiceCommandsEnabled = state
                        root.saveSettings()
                    }
                }

                SwitchRow {
                    label: "Notifications"
                    checked: Plasmoid.configuration.notificationsEnabled
                    onToggled: function(state) {
                        Plasmoid.configuration.notificationsEnabled = state
                        root.saveSettings()
                    }
                }

                SwitchRow {
                    label: "Auto-Punctuate"
                    checked: Plasmoid.configuration.autoPunctuate
                    onToggled: function(state) {
                        Plasmoid.configuration.autoPunctuate = state
                        root.saveSettings()
                    }
                }

                SwitchRow {
                    label: "Audio Feedback"
                    checked: Plasmoid.configuration.audioFeedbackEnabled
                    onToggled: function(state) {
                        Plasmoid.configuration.audioFeedbackEnabled = state
                        root.saveSettings()
                    }
                }

                SwitchRow {
                    label: "Dictation Formatting"
                    checked: Plasmoid.configuration.dictationFormattingEnabled
                    onToggled: function(state) {
                        Plasmoid.configuration.dictationFormattingEnabled = state
                        root.saveSettings()
                    }
                }

                SwitchRow {
                    label: "Mute Other Streams"
                    checked: Plasmoid.configuration.muteOtherStreams
                    onToggled: function(state) {
                        Plasmoid.configuration.muteOtherStreams = state
                        root.saveSettings()
                    }
                }

                Kirigami.Separator { Layout.fillWidth: true }

                // ----- Translate, VAD, Stream Interval -----
                ComboRow {
                    label: "Translate To"
                    model: translateTargets.map(function(t) { return t.label })
                    currentValue: {
                        var val = Plasmoid.configuration.translateTo || ""
                        for (var i = 0; i < translateTargets.length; i++) {
                            if (translateTargets[i].value === val) return translateTargets[i].label
                        }
                        return "Disabled"
                    }
                    onSelected: function(value) {
                        for (var i = 0; i < translateTargets.length; i++) {
                            if (translateTargets[i].label === value) {
                                Plasmoid.configuration.translateTo = translateTargets[i].value
                                root.saveSettings()
                                break
                            }
                        }
                    }
                }

                ComboRow {
                    label: "VAD Threshold"
                    model: vadOptions.map(function(v) { return v.label })
                    currentValue: {
                        var val = Plasmoid.configuration.vadThreshold || -40
                        for (var i = 0; i < vadOptions.length; i++) {
                            if (vadOptions[i].value === val) return vadOptions[i].label
                        }
                        return "-40 dB (default)"
                    }
                    onSelected: function(value) {
                        for (var i = 0; i < vadOptions.length; i++) {
                            if (vadOptions[i].label === value) {
                                Plasmoid.configuration.vadThreshold = vadOptions[i].value
                                root.saveSettings()
                                break
                            }
                        }
                    }
                }

                ComboRow {
                    label: "Stream Interval"
                    model: intervalOptions.map(function(v) { return v.label })
                    currentValue: {
                        var val = Plasmoid.configuration.streamInterval || 3.0
                        for (var i = 0; i < intervalOptions.length; i++) {
                            if (intervalOptions[i].value === val) return intervalOptions[i].label
                        }
                        return "3.0s (default)"
                    }
                    onSelected: function(value) {
                        for (var i = 0; i < intervalOptions.length; i++) {
                            if (intervalOptions[i].label === value) {
                                Plasmoid.configuration.streamInterval = intervalOptions[i].value
                                root.saveSettings()
                                break
                            }
                        }
                    }
                }

                Kirigami.Separator { Layout.fillWidth: true }

                // ----- Export History -----
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Kirigami.Units.smallSpacing

                    PlasmaComponents.Label {
                        text: "Export History:"
                        Layout.fillWidth: true
                    }

                    PlasmaComponents.Button {
                        text: "JSON"
                        onClicked: exportHistory("json")
                    }
                    PlasmaComponents.Button {
                        text: "Markdown"
                        onClicked: exportHistory("markdown")
                    }
                    PlasmaComponents.Button {
                        text: "SRT"
                        onClicked: exportHistory("srt")
                    }
                }

                Kirigami.Separator { Layout.fillWidth: true }

                // ----- Restart Server -----
                PlasmaComponents.Button {
                    Layout.fillWidth: true
                    text: "Restart Server"
                    icon.name: "system-reboot"

                    property bool restarting: false

                    onClicked: {
                        if (restarting) return
                        restarting = true
                        text = "Restarting server..."
                        restartServer(function() {
                            restarting = false
                            text = "Restart Server"
                        })
                    }
                }

                Item { Layout.preferredHeight: Kirigami.Units.smallSpacing }
            }
        }
    }

    function exportHistory(format) {
        var ext = format === "markdown" ? "md" : format
        var home = root.homeDir
        var filePath = home + "/whisper-history." + ext

        root.httpGet(root.serverUrl("/history/export?format=" + format + "&limit=50"),
            function(data, err) {
                if (data) {
                    var content = format === "json" ? JSON.stringify(data, null, 2) : String(data)
                    root.executable.exec("cat > " + filePath + " << 'EXPORTEOF'\n" + content + "\nEXPORTEOF")
                }
            })
    }

    function restartServer(callback) {
        var device = Plasmoid.configuration.device || "NPU"
        var home = root.homeDir

        var serverOverrideDir = home + "/.config/systemd/user/whisper-server.service.d"
        var serverOverride = "[Service]\nEnvironment=\"WHISPER_DEVICE=" + device + "\"\n"

        var cmd = "mkdir -p " + serverOverrideDir
        cmd += " && cat > " + serverOverrideDir + "/override.conf << 'SVREOF'\n" + serverOverride + "SVREOF"
        cmd += " && systemctl --user daemon-reload"
        cmd += " && systemctl --user restart whisper-server.service"

        root.executable.execCallback(cmd, function(stdout, stderr, exitCode) {
            if (exitCode === 0) {
                root.refreshStatus()
            }
            if (callback) callback()
        })
    }

    // ----- Reusable Components -----

    component SwitchRow: RowLayout {
        property string label
        property bool checked
        signal toggled(bool state)

        Layout.fillWidth: true
        spacing: Kirigami.Units.smallSpacing

        PlasmaComponents.Label {
            text: label
            Layout.fillWidth: true
        }
        PlasmaComponents.Switch {
            checked: parent.checked
            onToggled: parent.toggled(checked)
        }
    }

    component ComboRow: RowLayout {
        property string label
        property var model: []
        property string currentValue
        signal selected(string value)

        Layout.fillWidth: true
        spacing: Kirigami.Units.smallSpacing

        PlasmaComponents.Label {
            text: label
            Layout.fillWidth: true
        }
        PlasmaComponents.ComboBox {
            model: parent.model
            currentIndex: {
                var items = parent.model
                for (var i = 0; i < items.length; i++) {
                    if (items[i] === parent.currentValue) return i
                }
                return 0
            }
            onActivated: function(index) {
                parent.selected(parent.model[index])
            }
        }
    }

    component CollapsibleSection: ColumnLayout {
        property string title
        property alias content: contentLoader.sourceComponent
        property bool expanded: false

        Layout.fillWidth: true
        spacing: 0

        PlasmaComponents.ItemDelegate {
            Layout.fillWidth: true
            text: title
            icon.name: expanded ? "arrow-down" : "arrow-right"
            onClicked: expanded = !expanded
        }

        Loader {
            id: contentLoader
            Layout.fillWidth: true
            Layout.leftMargin: Kirigami.Units.largeSpacing
            visible: expanded
            active: expanded
        }
    }
}

import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import org.kde.kcmutils as KCM

KCM.SimpleKCM {
    id: configPage

    property alias cfg_serverHost: hostField.text
    property alias cfg_serverPort: portSpin.value
    property alias cfg_device: deviceCombo.currentIndex
    property alias cfg_backend: backendCombo.currentIndex
    property alias cfg_hotkey: hotkeyField.text
    property alias cfg_streamInterval: intervalSpin.value
    property alias cfg_hfOrg: hfOrgField.text
    property alias cfg_languageBuddyEnabled: buddySwitch.checked
    property alias cfg_languageBuddyBypass: bypassSwitch.checked
    property alias cfg_languageBuddyTimeout: timeoutSpin.value
    property alias cfg_voiceCommandsEnabled: voiceCommandsSwitch.checked
    property alias cfg_notificationsEnabled: notificationsSwitch.checked
    property alias cfg_autoPunctuate: autoPunctuateSwitch.checked
    property alias cfg_audioFeedbackEnabled: audioFeedbackSwitch.checked
    property alias cfg_dictationFormattingEnabled: formattingSwitch.checked
    property alias cfg_muteOtherStreams: muteStreamsSwitch.checked

    Kirigami.FormLayout {
        // Server Connection
        Kirigami.Separator {
            Kirigami.FormData.isSection: true
            Kirigami.FormData.label: "Server Connection"
        }

        QQC2.TextField {
            id: hostField
            Kirigami.FormData.label: "Host:"
            placeholderText: "127.0.0.1"
        }

        QQC2.SpinBox {
            id: portSpin
            Kirigami.FormData.label: "Port:"
            from: 1024
            to: 65535
        }

        // Server Options
        Kirigami.Separator {
            Kirigami.FormData.isSection: true
            Kirigami.FormData.label: "Server Options"
        }

        QQC2.ComboBox {
            id: deviceCombo
            Kirigami.FormData.label: "Device:"
            model: ["NPU", "CPU", "GPU"]
        }

        QQC2.ComboBox {
            id: backendCombo
            Kirigami.FormData.label: "Backend:"
            model: ["openvino", "whisper-cpp"]
        }

        QQC2.TextField {
            id: hotkeyField
            Kirigami.FormData.label: "Hotkey (evdev):"
            placeholderText: "KEY_RIGHTCTRL"
        }

        QQC2.SpinBox {
            id: intervalSpin
            Kirigami.FormData.label: "Stream Interval (s):"
            from: 5
            to: 300
            stepSize: 5
            property int decimals: 1
            property real realValue: value / 10.0
            value: cfg_streamInterval * 10

            textFromValue: function(value, locale) {
                return (value / 10.0).toFixed(1)
            }
            valueFromText: function(text, locale) {
                return Math.round(parseFloat(text) * 10)
            }
        }

        QQC2.TextField {
            id: hfOrgField
            Kirigami.FormData.label: "HuggingFace Org:"
            placeholderText: "OpenVINO"
        }

        // Language Buddy
        Kirigami.Separator {
            Kirigami.FormData.isSection: true
            Kirigami.FormData.label: "Language Buddy"
        }

        QQC2.Switch {
            id: buddySwitch
            Kirigami.FormData.label: "Enable Language Buddy:"
        }

        QQC2.Switch {
            id: bypassSwitch
            Kirigami.FormData.label: "Bypass mode:"
        }

        QQC2.SpinBox {
            id: timeoutSpin
            Kirigami.FormData.label: "Overlay timeout (s):"
            from: 0
            to: 120
        }

        // Features
        Kirigami.Separator {
            Kirigami.FormData.isSection: true
            Kirigami.FormData.label: "Features"
        }

        QQC2.Switch {
            id: voiceCommandsSwitch
            Kirigami.FormData.label: "Voice Commands:"
        }

        QQC2.Switch {
            id: notificationsSwitch
            Kirigami.FormData.label: "Notifications:"
        }

        QQC2.Switch {
            id: autoPunctuateSwitch
            Kirigami.FormData.label: "Auto-Punctuate:"
        }

        QQC2.Switch {
            id: audioFeedbackSwitch
            Kirigami.FormData.label: "Audio Feedback:"
        }

        QQC2.Switch {
            id: formattingSwitch
            Kirigami.FormData.label: "Dictation Formatting:"
        }

        QQC2.Switch {
            id: muteStreamsSwitch
            Kirigami.FormData.label: "Mute Other Streams:"
        }
    }
}

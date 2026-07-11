import QtQuick
import QtQuick.Layouts
import org.kde.plasma.plasmoid
import org.kde.plasma.plasma5support as P5Support
import org.kde.kirigami as Kirigami

PlasmoidItem {
    id: root

    property string homeDir: ""
    property string serverHost: Plasmoid.configuration.serverHost || "127.0.0.1"
    property int serverPort: Plasmoid.configuration.serverPort || 5000
    property bool connected: false
    property string currentModel: ""
    property string currentLlmModel: ""
    property var localModels: []
    property var localLlmModels: []
    property var remoteModels: []
    property var remoteLlmModels: []

    switchWidth: Kirigami.Units.gridUnit * 20
    switchHeight: Kirigami.Units.gridUnit * 30

    toolTipMainText: "Whisper NPU"
    toolTipSubText: connected ? "Connected | Model: " + currentModel : "Disconnected"

    Plasmoid.icon: "audio-input-microphone"

    compactRepresentation: CompactRepresentation {}

    fullRepresentation: FullRepresentation {
        serverHost: root.serverHost
        serverPort: root.serverPort
        connected: root.connected
        currentModel: root.currentModel
        currentLlmModel: root.currentLlmModel
        localModels: root.localModels
        localLlmModels: root.localLlmModels
        remoteModels: root.remoteModels
        remoteLlmModels: root.remoteLlmModels
    }

    function serverUrl(path) {
        return "http://" + serverHost + ":" + serverPort + path
    }

    function httpGet(url, callback) {
        var xhr = new XMLHttpRequest()
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status === 200) {
                    try {
                        callback(JSON.parse(xhr.responseText), null)
                    } catch (e) {
                        callback(null, e.message)
                    }
                } else {
                    callback(null, "HTTP " + xhr.status)
                }
            }
        }
        xhr.open("GET", url)
        xhr.timeout = 5000
        xhr.send()
    }

    function httpRequest(method, url, body, callback) {
        var xhr = new XMLHttpRequest()
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status === 200) {
                    try {
                        callback(JSON.parse(xhr.responseText), null)
                    } catch (e) {
                        callback(null, e.message)
                    }
                } else {
                    callback(null, "HTTP " + xhr.status)
                }
            }
        }
        xhr.open(method, url)
        xhr.setRequestHeader("Content-Type", "application/json")
        xhr.timeout = 10000
        xhr.send(JSON.stringify(body))
    }

    function refreshStatus() {
        httpGet(serverUrl("/health"), function(data, err) {
            if (data) {
                connected = true
                currentModel = data.model || ""
            } else {
                connected = false
                currentModel = ""
            }
        })

        httpGet(serverUrl("/llm/models"), function(data, err) {
            if (data) {
                currentLlmModel = data.current || ""
            }
        })
    }

    function refreshRemoteModels() {
        var org = Plasmoid.configuration.hfOrg || "OpenVINO"
        var url = "https://huggingface.co/api/models?author=" + org + "&search=whisper&sort=downloads&direction=-1&limit=50"
        httpGet(url, function(data, err) {
            if (data && Array.isArray(data)) {
                var models = []
                for (var i = 0; i < data.length; i++) {
                    var m = data[i]
                    if (m.id && m.id.toLowerCase().indexOf("whisper") >= 0) {
                        models.push({
                            id: m.id,
                            name: m.id.split("/").pop(),
                            downloads: m.downloads || 0
                        })
                    }
                }
                remoteModels = models
            }
        })

        var llmUrl = "https://huggingface.co/api/models?author=" + org + "&search=int4-ov&sort=downloads&direction=-1&limit=50"
        httpGet(llmUrl, function(data, err) {
            if (data && Array.isArray(data)) {
                var models = []
                for (var i = 0; i < data.length; i++) {
                    var m = data[i]
                    if (m.id && m.id.toLowerCase().indexOf("int4-ov") >= 0) {
                        models.push({
                            id: m.id,
                            name: m.id.split("/").pop(),
                            downloads: m.downloads || 0
                        })
                    }
                }
                remoteLlmModels = models
            }
        })
    }

    function saveSettings() {
        var settings = {
            "server-host": Plasmoid.configuration.serverHost,
            "server-port": Plasmoid.configuration.serverPort,
            "device": Plasmoid.configuration.device,
            "backend": Plasmoid.configuration.backend,
            "hotkey": Plasmoid.configuration.hotkey,
            "language": Plasmoid.configuration.language,
            "recall-key": Plasmoid.configuration.recallKey,
            "hf-org": Plasmoid.configuration.hfOrg,
            "language-buddy-enabled": Plasmoid.configuration.languageBuddyEnabled,
            "language-buddy-bypass": Plasmoid.configuration.languageBuddyBypass,
            "language-buddy-timeout": Plasmoid.configuration.languageBuddyTimeout,
            "voice-commands-enabled": Plasmoid.configuration.voiceCommandsEnabled,
            "notifications-enabled": Plasmoid.configuration.notificationsEnabled,
            "auto-punctuate": Plasmoid.configuration.autoPunctuate,
            "audio-feedback-enabled": Plasmoid.configuration.audioFeedbackEnabled,
            "dictation-formatting-enabled": Plasmoid.configuration.dictationFormattingEnabled,
            "vad-threshold": Plasmoid.configuration.vadThreshold,
            "stream-interval": Plasmoid.configuration.streamInterval,
            "translate-to": Plasmoid.configuration.translateTo,
            "mute-other-streams": Plasmoid.configuration.muteOtherStreams
        }

        var configDir = homeDir + "/.config/whisper-npu"
        var configPath = configDir + "/settings.json"
        var content = JSON.stringify(settings, null, 2)

        executable.exec("mkdir -p " + configDir + " && cat > " + configPath + " << 'SETTINGSEOF'\n" + content + "\nSETTINGSEOF")
    }

    P5Support.DataSource {
        id: executable
        engine: "executable"
        connectedSources: []

        function exec(cmd) {
            connectSource(cmd)
        }

        function execCallback(cmd, callback) {
            var handler = function(source, data) {
                if (source === cmd) {
                    executable.disconnectSource(cmd)
                    executable.dataChanged.disconnect(handler)
                    if (callback) callback(data["stdout"] || "", data["stderr"] || "", data["exit code"] || 0)
                }
            }
            dataChanged.connect(handler)
            connectSource(cmd)
        }

        onNewData: (source, data) => {
            disconnectSource(source)
        }
    }

    function refreshLocalModels() {
        var modelsDir = homeDir + "/.whisper/models"
        executable.execCallback("ls -1 " + modelsDir + " 2>/dev/null", function(stdout, stderr, exitCode) {
            if (exitCode === 0 && stdout.trim()) {
                localModels = stdout.trim().split("\n").filter(function(n) {
                    return n && !n.startsWith(".")
                }).sort()
            } else {
                localModels = []
            }
        })

        var llmDir = homeDir + "/.whisper/llm-models"
        executable.execCallback("ls -1 " + llmDir + " 2>/dev/null", function(stdout, stderr, exitCode) {
            if (exitCode === 0 && stdout.trim()) {
                localLlmModels = stdout.trim().split("\n").filter(function(n) {
                    return n && !n.startsWith(".")
                }).sort()
            } else {
                localLlmModels = []
            }
        })
    }

    Component.onCompleted: {
        executable.execCallback("echo $HOME", function(stdout) {
            homeDir = stdout.trim()
            refreshStatus()
            refreshLocalModels()
            refreshRemoteModels()
        })
    }
}

import QtQuick
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

MouseArea {
    id: compactRoot

    property bool wasExpanded: false

    acceptedButtons: Qt.LeftButton
    hoverEnabled: true

    onPressed: wasExpanded = root.expanded
    onClicked: {
        root.expanded = !wasExpanded
        if (root.expanded) {
            root.refreshStatus()
            root.refreshLocalModels()
        }
    }

    Kirigami.Icon {
        anchors.fill: parent
        source: Qt.resolvedUrl("../icons/whisper-npu.svg")
        active: compactRoot.containsMouse
    }
}

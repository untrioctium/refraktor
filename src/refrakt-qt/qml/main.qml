import QtQuick.Controls.Fusion

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    visible: true
    width: 1200
    height: 800
    title: "Test Window"

    menuBar: MenuBar {

        Menu {
            topPadding: 0
            bottomPadding: 0
            
            title: qsTr("&File")
            Action { text: qsTr("&New")}
            Action { text: qsTr("&Open")}
            Action { text: qsTr("&Save")}
            Action { text: qsTr("&Save As")}
            Action { text: qsTr("&Close")}
            MenuSeparator {}
            Action { text: qsTr("&Quit")}
        }
    }

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        Rectangle {
            color: "#1e1e1e"
            SplitView.preferredWidth: window.width / 4

            RowLayout {
                id: test

                TextField {
                    id: textField
                    text: dial.value
                    onTextChanged: dial.value = text
                }
                Dial {
                    id: dial
                    implicitWidth: textField.implicitHeight
                    implicitHeight: textField.implicitHeight
                    value: textField.value
                }
                Label {
                    text: "Linear"
                }
            }
        }

        Rectangle {
            SplitView.preferredWidth: window.width * 3 / 4
            color: "black"
        }
    }
}
import QtQuick.Controls.Fusion

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Refrakt

ApplicationWindow {
    id: window
    visible: true
    width: 1200
    height: 800
    title: "Test Window"

    menuBar: MenuBar {
        topPadding: 0
        bottomPadding: 0

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

            ListView {
                id: flameList
                anchors.fill: parent
                anchors.margins: 4
                interactive: false
                clip: true
                spacing: 4

                WheelHandler {
                    onWheel: (event) => {
                        flameList.contentY = Math.max(0, 
                            Math.min(flameList.contentHeight - flameList.height,
                                flameList.contentY - event.angleDelta.y))
                    }
                }

                model: ListModel {
                    id: flamesModel
                }

                property var requestId: null
                Component.onCompleted: {
                    flameList.requestId = FlameDirectoryService.listFlames()
                }

                Connections {
                    target: FlameDirectoryService
                    function onFlamesListed(id, flames) {
                        if (flameList.requestId === id) {
                            flamesModel.clear()
                            for (const flame of flames) {
                                flamesModel.append({ flameName: flame })
                            }
                            flameList.requestId = null
                        }
                    }
                }

                delegate: Item {
                    width: flameList.width
                    height: width * 3 / 4  // 4:3 aspect ratio

                    Rectangle {
                        anchors.fill: parent
                        anchors.margins: 2
                        color: "black"
                        border.color: flamePreview.source === model.flameName ? "#0078d4" : "transparent"
                        border.width: 2

                        FlamePreview {
                            anchors.fill: parent
                            anchors.margins: 2
                            source: model.flameName
                            quality: 50
                            denoise: true
                            upscale: true
                            maxRenderMillis: 50

                            BusyIndicator {
                                anchors.centerIn: parent
                                running: parent.status === FlamePreview.Loading
                            }
                        }

                        Text {
                            anchors.bottom: parent.bottom
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.margins: 6
                            text: model.flameName
                            color: "white"
                            style: Text.Outline
                            styleColor: "black"
                            font.pixelSize: 11
                            elide: Text.ElideMiddle
                        }

                        MouseArea {
                            anchors.fill: parent
                            onClicked: {
                                flamePreview.source = model.flameName
                            }

                            onPressed: {
                                flameList.cancelFlick();
                            }
                        }
                    }
                }

                ScrollBar.vertical: ScrollBar {}
            }
        }

        Rectangle {
            SplitView.preferredWidth: window.width * 3 / 4
            color: "black"

            FlamePreview {
                id: flamePreview
                anchors.fill: parent
                source: ""
                quality: 1000
                denoise: true
                upscale: false
                maxRenderMillis: 34
                
                BusyIndicator {
                    anchors.centerIn: parent
                    running: parent.status === FlamePreview.Loading
                }
            }
        }
    }
}
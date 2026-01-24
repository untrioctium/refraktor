#pragma once

#include <QObject>
#include <QAbstractListModel>
#include <qqmlintegration.h>

#include <librefrakt/flame_types.hpp>

// Forward declarations
class FlameHandle;
class XformHandle;
class VLinkHandle;
class VariationHandle;
class XformListModel;
class VLinkListModel;
class VariationListModel;
class ParameterListModel;

// ============================================================================
// ParameterListModel - exposes variation parameters
// ============================================================================

class ParameterListModel : public QAbstractListModel {
    Q_OBJECT

public:
    enum Roles {
        NameRole = Qt::UserRole + 1,
        ValueRole
    };

    ParameterListModel(rfkt::flame* flame, int xformIndex, int vlinkIndex,
                       const QString& variationName, QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role) override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    QHash<int, QByteArray> roleNames() const override;

    Q_INVOKABLE void refresh();

private:
    rfkt::vardata* vardata() const;
    std::vector<std::string> parameterNames() const;

    rfkt::flame* flame_;
    int xformIndex_;
    int vlinkIndex_;
    QString variationName_;
};

// ============================================================================
// VariationHandle - non-owning reference to a variation
// ============================================================================

class VariationHandle : public QObject {
    Q_OBJECT
    QML_ELEMENT
    QML_UNCREATABLE("VariationHandle is created by VLinkHandle")

    Q_PROPERTY(QString name READ name CONSTANT)
    Q_PROPERTY(double weight READ weight WRITE setWeight NOTIFY weightChanged)
    Q_PROPERTY(ParameterListModel* parameters READ parameters CONSTANT)

public:
    VariationHandle(rfkt::flame* flame, int xformIndex, int vlinkIndex,
                    const QString& variationName, QObject* parent = nullptr);

    QString name() const { return variationName_; }
    double weight() const;
    void setWeight(double w);

    ParameterListModel* parameters() { return &parametersModel_; }

    Q_INVOKABLE double getParameter(const QString& name) const;
    Q_INVOKABLE void setParameter(const QString& name, double value);

signals:
    void weightChanged();
    void parameterChanged(const QString& name);

private:
    rfkt::vardata* vardata() const;

    rfkt::flame* flame_;
    int xformIndex_;
    int vlinkIndex_;
    QString variationName_;
    ParameterListModel parametersModel_;
};

// ============================================================================
// VariationListModel - exposes variations in a vlink
// ============================================================================

class VariationListModel : public QAbstractListModel {
    Q_OBJECT

public:
    enum Roles {
        NameRole = Qt::UserRole + 1,
        HandleRole
    };

    VariationListModel(rfkt::flame* flame, int xformIndex, int vlinkIndex,
                       QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    Q_INVOKABLE void refresh();
    Q_INVOKABLE VariationHandle* get(const QString& name);

private:
    rfkt::vlink* vlink() const;
    std::vector<std::string> variationNames() const;
    VariationHandle* getOrCreateHandle(const QString& name) const;

    rfkt::flame* flame_;
    int xformIndex_;
    int vlinkIndex_;
    mutable QMap<QString, VariationHandle*> handleCache_;
};

// ============================================================================
// VLinkHandle - non-owning reference to a vlink
// ============================================================================

class VLinkHandle : public QObject {
    Q_OBJECT
    QML_ELEMENT
    QML_UNCREATABLE("VLinkHandle is created by XformHandle")

    Q_PROPERTY(int index READ index CONSTANT)

    // Transform properties
    Q_PROPERTY(double transformA READ transformA WRITE setTransformA NOTIFY transformChanged)
    Q_PROPERTY(double transformB READ transformB WRITE setTransformB NOTIFY transformChanged)
    Q_PROPERTY(double transformC READ transformC WRITE setTransformC NOTIFY transformChanged)
    Q_PROPERTY(double transformD READ transformD WRITE setTransformD NOTIFY transformChanged)
    Q_PROPERTY(double transformE READ transformE WRITE setTransformE NOTIFY transformChanged)
    Q_PROPERTY(double transformF READ transformF WRITE setTransformF NOTIFY transformChanged)

    // Modifier properties
    Q_PROPERTY(double modX READ modX WRITE setModX NOTIFY modifiersChanged)
    Q_PROPERTY(double modY READ modY WRITE setModY NOTIFY modifiersChanged)
    Q_PROPERTY(double modScale READ modScale WRITE setModScale NOTIFY modifiersChanged)
    Q_PROPERTY(double modRotate READ modRotate WRITE setModRotate NOTIFY modifiersChanged)

    Q_PROPERTY(VariationListModel* variations READ variations CONSTANT)

public:
    VLinkHandle(rfkt::flame* flame, int xformIndex, int vlinkIndex,
                QObject* parent = nullptr);

    int index() const { return vlinkIndex_; }

    // Transform accessors
    double transformA() const;
    double transformB() const;
    double transformC() const;
    double transformD() const;
    double transformE() const;
    double transformF() const;

    void setTransformA(double v);
    void setTransformB(double v);
    void setTransformC(double v);
    void setTransformD(double v);
    void setTransformE(double v);
    void setTransformF(double v);

    // Modifier accessors
    double modX() const;
    double modY() const;
    double modScale() const;
    double modRotate() const;

    void setModX(double v);
    void setModY(double v);
    void setModScale(double v);
    void setModRotate(double v);

    VariationListModel* variations() { return &variationsModel_; }

signals:
    void transformChanged();
    void modifiersChanged();

private:
    rfkt::vlink* vlink() const;

    rfkt::flame* flame_;
    int xformIndex_;
    int vlinkIndex_;
    VariationListModel variationsModel_;
};

// ============================================================================
// VLinkListModel - exposes vlinks in an xform
// ============================================================================

class VLinkListModel : public QAbstractListModel {
    Q_OBJECT

public:
    enum Roles {
        IndexRole = Qt::UserRole + 1,
        HandleRole
    };

    VLinkListModel(rfkt::flame* flame, int xformIndex, QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    Q_INVOKABLE void refresh();
    Q_INVOKABLE VLinkHandle* get(int index);

private:
    rfkt::xform* xform() const;
    VLinkHandle* getOrCreateHandle(int index) const;

    rfkt::flame* flame_;
    int xformIndex_;
    mutable std::vector<VLinkHandle*> handleCache_;
};

// ============================================================================
// XformHandle - non-owning reference to an xform
// ============================================================================

class XformHandle : public QObject {
    Q_OBJECT
    QML_ELEMENT
    QML_UNCREATABLE("XformHandle is created by FlameHandle")

    Q_PROPERTY(int index READ index CONSTANT)
    Q_PROPERTY(bool isFinal READ isFinal CONSTANT)

    Q_PROPERTY(double weight READ weight WRITE setWeight NOTIFY weightChanged)
    Q_PROPERTY(double color READ color WRITE setColor NOTIFY colorChanged)
    Q_PROPERTY(double colorSpeed READ colorSpeed WRITE setColorSpeed NOTIFY colorSpeedChanged)
    Q_PROPERTY(double opacity READ opacity WRITE setOpacity NOTIFY opacityChanged)

    Q_PROPERTY(VLinkListModel* vlinks READ vlinks CONSTANT)

public:
    XformHandle(rfkt::flame* flame, int index, QObject* parent = nullptr);

    int index() const { return index_; }
    bool isFinal() const { return index_ == -1; }

    double weight() const;
    double color() const;
    double colorSpeed() const;
    double opacity() const;

    void setWeight(double w);
    void setColor(double c);
    void setColorSpeed(double cs);
    void setOpacity(double o);

    VLinkListModel* vlinks() { return &vlinksModel_; }

signals:
    void weightChanged();
    void colorChanged();
    void colorSpeedChanged();
    void opacityChanged();

private:
    rfkt::xform* xform() const;

    rfkt::flame* flame_;
    int index_;  // -1 for final xform
    VLinkListModel vlinksModel_;
};

// ============================================================================
// XformListModel - exposes xforms in a flame
// ============================================================================

class XformListModel : public QAbstractListModel {
    Q_OBJECT

public:
    enum Roles {
        IndexRole = Qt::UserRole + 1,
        IsFinalRole,
        HandleRole
    };

    explicit XformListModel(rfkt::flame* flame, QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    Q_INVOKABLE void refresh();
    Q_INVOKABLE XformHandle* get(int index);

    void setFlame(rfkt::flame* flame);

private:
    XformHandle* getOrCreateHandle(int index) const;
    int xformIndexFromRow(int row) const;

    rfkt::flame* flame_;
    mutable std::vector<XformHandle*> handleCache_;  // regular xforms
    mutable XformHandle* finalXformHandle_ = nullptr;
};

// ============================================================================
// FlameHandle - top-level handle to a flame
// ============================================================================

class FlameHandle : public QObject {
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(QString name READ name WRITE setName NOTIFY nameChanged)

    // Camera properties
    Q_PROPERTY(double centerX READ centerX WRITE setCenterX NOTIFY cameraChanged)
    Q_PROPERTY(double centerY READ centerY WRITE setCenterY NOTIFY cameraChanged)
    Q_PROPERTY(double scale READ scale WRITE setScale NOTIFY cameraChanged)
    Q_PROPERTY(double rotate READ rotate WRITE setRotate NOTIFY cameraChanged)

    // Render properties
    Q_PROPERTY(double gamma READ gamma WRITE setGamma NOTIFY renderSettingsChanged)
    Q_PROPERTY(double brightness READ brightness WRITE setBrightness NOTIFY renderSettingsChanged)
    Q_PROPERTY(double vibrancy READ vibrancy WRITE setVibrancy NOTIFY renderSettingsChanged)
    Q_PROPERTY(double highlightPower READ highlightPower WRITE setHighlightPower NOTIFY renderSettingsChanged)
    Q_PROPERTY(double gammaThreshold READ gammaThreshold WRITE setGammaThreshold NOTIFY renderSettingsChanged)

    // Palette modifiers
    Q_PROPERTY(double modHue READ modHue WRITE setModHue NOTIFY paletteChanged)
    Q_PROPERTY(double modSat READ modSat WRITE setModSat NOTIFY paletteChanged)
    Q_PROPERTY(double modVal READ modVal WRITE setModVal NOTIFY paletteChanged)

    Q_PROPERTY(XformListModel* xforms READ xforms CONSTANT)
    Q_PROPERTY(bool hasFinalXform READ hasFinalXform NOTIFY structureChanged)

public:
    explicit FlameHandle(QObject* parent = nullptr);
    explicit FlameHandle(rfkt::flame* flame, QObject* parent = nullptr);

    void setFlame(rfkt::flame* flame);
    rfkt::flame* flame() const { return flame_; }

    QString name() const;
    void setName(const QString& name);

    // Camera
    double centerX() const;
    double centerY() const;
    double scale() const;
    double rotate() const;

    void setCenterX(double v);
    void setCenterY(double v);
    void setScale(double v);
    void setRotate(double v);

    // Render settings
    double gamma() const;
    double brightness() const;
    double vibrancy() const;
    double highlightPower() const;
    double gammaThreshold() const;

    void setGamma(double v);
    void setBrightness(double v);
    void setVibrancy(double v);
    void setHighlightPower(double v);
    void setGammaThreshold(double v);

    // Palette
    double modHue() const;
    double modSat() const;
    double modVal() const;

    void setModHue(double v);
    void setModSat(double v);
    void setModVal(double v);

    XformListModel* xforms() { return &xformsModel_; }
    bool hasFinalXform() const;

    // Path-based parameter access
    Q_INVOKABLE QVariant getParameter(const QString& path) const;
    Q_INVOKABLE bool setParameter(const QString& path, double value);

signals:
    void nameChanged();
    void cameraChanged();
    void renderSettingsChanged();
    void paletteChanged();
    void structureChanged();

private:
    rfkt::flame* flame_ = nullptr;
    XformListModel xformsModel_;
};

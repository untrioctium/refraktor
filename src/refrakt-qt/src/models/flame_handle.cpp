#include "flame_handle.hpp"

// ============================================================================
// ParameterListModel
// ============================================================================

ParameterListModel::ParameterListModel(rfkt::flame* flame, int xformIndex,
                                       int vlinkIndex, const QString& variationName,
                                       QObject* parent)
    : QAbstractListModel(parent)
    , flame_(flame)
    , xformIndex_(xformIndex)
    , vlinkIndex_(vlinkIndex)
    , variationName_(variationName)
{
}

rfkt::vardata* ParameterListModel::vardata() const {
    if (!flame_) return nullptr;
    auto* xf = flame_->get_xform(xformIndex_);
    if (!xf || vlinkIndex_ < 0 || vlinkIndex_ >= static_cast<int>(xf->vchain.size()))
        return nullptr;
    auto& vl = xf->vchain[vlinkIndex_];
    if (!vl.has_variation(variationName_.toStdString()))
        return nullptr;
    return &vl[variationName_.toStdString()];
}

std::vector<std::string> ParameterListModel::parameterNames() const {
    std::vector<std::string> names;
    if (auto* vd = vardata()) {
        for (const auto& [name, _] : *vd) {
            names.push_back(name);
        }
    }
    return names;
}

int ParameterListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    auto* vd = vardata();
    if (!vd) return 0;
    // Count parameters (excluding weight which is handled by VariationHandle)
    int count = 0;
    for (const auto& [_, __] : *vd) {
        ++count;
    }
    return count;
}

QVariant ParameterListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) return {};
    
    auto names = parameterNames();
    if (index.row() < 0 || index.row() >= static_cast<int>(names.size()))
        return {};
    
    const auto& name = names[index.row()];
    auto* vd = vardata();
    if (!vd) return {};

    switch (role) {
    case NameRole:
        return QString::fromStdString(name);
    case ValueRole:
        return (*vd)[name].t0;
    default:
        return {};
    }
}

bool ParameterListModel::setData(const QModelIndex& index, const QVariant& value, int role) {
    if (!index.isValid() || role != ValueRole) return false;
    
    auto names = parameterNames();
    if (index.row() < 0 || index.row() >= static_cast<int>(names.size()))
        return false;
    
    auto* vd = vardata();
    if (!vd) return false;

    const auto& name = names[index.row()];
    (*vd)[name].t0 = value.toDouble();
    emit dataChanged(index, index, {ValueRole});
    return true;
}

Qt::ItemFlags ParameterListModel::flags(const QModelIndex& index) const {
    if (!index.isValid()) return Qt::NoItemFlags;
    return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsEditable;
}

QHash<int, QByteArray> ParameterListModel::roleNames() const {
    return {
        {NameRole, "name"},
        {ValueRole, "value"}
    };
}

void ParameterListModel::refresh() {
    beginResetModel();
    endResetModel();
}

// ============================================================================
// VariationHandle
// ============================================================================

VariationHandle::VariationHandle(rfkt::flame* flame, int xformIndex,
                                 int vlinkIndex, const QString& variationName,
                                 QObject* parent)
    : QObject(parent)
    , flame_(flame)
    , xformIndex_(xformIndex)
    , vlinkIndex_(vlinkIndex)
    , variationName_(variationName)
    , parametersModel_(flame, xformIndex, vlinkIndex, variationName, this)
{
}

rfkt::vardata* VariationHandle::vardata() const {
    if (!flame_) return nullptr;
    auto* xf = flame_->get_xform(xformIndex_);
    if (!xf || vlinkIndex_ < 0 || vlinkIndex_ >= static_cast<int>(xf->vchain.size()))
        return nullptr;
    auto& vl = xf->vchain[vlinkIndex_];
    if (!vl.has_variation(variationName_.toStdString()))
        return nullptr;
    return &vl[variationName_.toStdString()];
}

double VariationHandle::weight() const {
    if (auto* vd = vardata()) return vd->weight.t0;
    return 0.0;
}

void VariationHandle::setWeight(double w) {
    if (auto* vd = vardata()) {
        vd->weight.t0 = w;
        emit weightChanged();
    }
}

double VariationHandle::getParameter(const QString& name) const {
    if (auto* vd = vardata()) {
        if (vd->has_parameter(name.toStdString())) {
            return (*vd)[name.toStdString()].t0;
        }
    }
    return 0.0;
}

void VariationHandle::setParameter(const QString& name, double value) {
    if (auto* vd = vardata()) {
        if (vd->has_parameter(name.toStdString())) {
            (*vd)[name.toStdString()].t0 = value;
            emit parameterChanged(name);
        }
    }
}

// ============================================================================
// VariationListModel
// ============================================================================

VariationListModel::VariationListModel(rfkt::flame* flame, int xformIndex,
                                       int vlinkIndex, QObject* parent)
    : QAbstractListModel(parent)
    , flame_(flame)
    , xformIndex_(xformIndex)
    , vlinkIndex_(vlinkIndex)
{
}

rfkt::vlink* VariationListModel::vlink() const {
    if (!flame_) return nullptr;
    auto* xf = flame_->get_xform(xformIndex_);
    if (!xf || vlinkIndex_ < 0 || vlinkIndex_ >= static_cast<int>(xf->vchain.size()))
        return nullptr;
    return &xf->vchain[vlinkIndex_];
}

std::vector<std::string> VariationListModel::variationNames() const {
    std::vector<std::string> names;
    if (auto* vl = vlink()) {
        for (const auto& [name, _] : *vl) {
            names.push_back(name);
        }
    }
    return names;
}

int VariationListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    auto* vl = vlink();
    return vl ? static_cast<int>(vl->size_variations()) : 0;
}

QVariant VariationListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) return {};
    
    auto names = variationNames();
    if (index.row() < 0 || index.row() >= static_cast<int>(names.size()))
        return {};

    switch (role) {
    case NameRole:
        return QString::fromStdString(names[index.row()]);
    case HandleRole:
        return QVariant::fromValue(getOrCreateHandle(QString::fromStdString(names[index.row()])));
    default:
        return {};
    }
}

QHash<int, QByteArray> VariationListModel::roleNames() const {
    return {
        {NameRole, "name"},
        {HandleRole, "handle"}
    };
}

VariationHandle* VariationListModel::getOrCreateHandle(const QString& name) const {
    auto it = handleCache_.find(name);
    if (it != handleCache_.end()) {
        return it.value();
    }
    
    auto* handle = new VariationHandle(flame_, xformIndex_, vlinkIndex_, name,
                                       const_cast<VariationListModel*>(this));
    handleCache_.insert(name, handle);
    return handle;
}

void VariationListModel::refresh() {
    beginResetModel();
    qDeleteAll(handleCache_);
    handleCache_.clear();
    endResetModel();
}

VariationHandle* VariationListModel::get(const QString& name) {
    return getOrCreateHandle(name);
}

// ============================================================================
// VLinkHandle
// ============================================================================

VLinkHandle::VLinkHandle(rfkt::flame* flame, int xformIndex, int vlinkIndex,
                         QObject* parent)
    : QObject(parent)
    , flame_(flame)
    , xformIndex_(xformIndex)
    , vlinkIndex_(vlinkIndex)
    , variationsModel_(flame, xformIndex, vlinkIndex, this)
{
}

rfkt::vlink* VLinkHandle::vlink() const {
    if (!flame_) return nullptr;
    auto* xf = flame_->get_xform(xformIndex_);
    if (!xf || vlinkIndex_ < 0 || vlinkIndex_ >= static_cast<int>(xf->vchain.size()))
        return nullptr;
    return &xf->vchain[vlinkIndex_];
}

double VLinkHandle::transformA() const {
    if (auto* vl = vlink()) return vl->transform.a.t0;
    return 0.0;
}

double VLinkHandle::transformB() const {
    if (auto* vl = vlink()) return vl->transform.b.t0;
    return 0.0;
}

double VLinkHandle::transformC() const {
    if (auto* vl = vlink()) return vl->transform.c.t0;
    return 0.0;
}

double VLinkHandle::transformD() const {
    if (auto* vl = vlink()) return vl->transform.d.t0;
    return 0.0;
}

double VLinkHandle::transformE() const {
    if (auto* vl = vlink()) return vl->transform.e.t0;
    return 0.0;
}

double VLinkHandle::transformF() const {
    if (auto* vl = vlink()) return vl->transform.f.t0;
    return 0.0;
}

void VLinkHandle::setTransformA(double v) {
    if (auto* vl = vlink()) { vl->transform.a.t0 = v; emit transformChanged(); }
}

void VLinkHandle::setTransformB(double v) {
    if (auto* vl = vlink()) { vl->transform.b.t0 = v; emit transformChanged(); }
}

void VLinkHandle::setTransformC(double v) {
    if (auto* vl = vlink()) { vl->transform.c.t0 = v; emit transformChanged(); }
}

void VLinkHandle::setTransformD(double v) {
    if (auto* vl = vlink()) { vl->transform.d.t0 = v; emit transformChanged(); }
}

void VLinkHandle::setTransformE(double v) {
    if (auto* vl = vlink()) { vl->transform.e.t0 = v; emit transformChanged(); }
}

void VLinkHandle::setTransformF(double v) {
    if (auto* vl = vlink()) { vl->transform.f.t0 = v; emit transformChanged(); }
}

double VLinkHandle::modX() const {
    if (auto* vl = vlink()) return vl->mod_x.t0;
    return 0.0;
}

double VLinkHandle::modY() const {
    if (auto* vl = vlink()) return vl->mod_y.t0;
    return 0.0;
}

double VLinkHandle::modScale() const {
    if (auto* vl = vlink()) return vl->mod_scale.t0;
    return 1.0;
}

double VLinkHandle::modRotate() const {
    if (auto* vl = vlink()) return vl->mod_rotate.t0;
    return 0.0;
}

void VLinkHandle::setModX(double v) {
    if (auto* vl = vlink()) { vl->mod_x.t0 = v; emit modifiersChanged(); }
}

void VLinkHandle::setModY(double v) {
    if (auto* vl = vlink()) { vl->mod_y.t0 = v; emit modifiersChanged(); }
}

void VLinkHandle::setModScale(double v) {
    if (auto* vl = vlink()) { vl->mod_scale.t0 = v; emit modifiersChanged(); }
}

void VLinkHandle::setModRotate(double v) {
    if (auto* vl = vlink()) { vl->mod_rotate.t0 = v; emit modifiersChanged(); }
}

// ============================================================================
// VLinkListModel
// ============================================================================

VLinkListModel::VLinkListModel(rfkt::flame* flame, int xformIndex, QObject* parent)
    : QAbstractListModel(parent)
    , flame_(flame)
    , xformIndex_(xformIndex)
{
}

rfkt::xform* VLinkListModel::xform() const {
    if (!flame_) return nullptr;
    return flame_->get_xform(xformIndex_);
}

int VLinkListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    auto* xf = xform();
    return xf ? static_cast<int>(xf->vchain.size()) : 0;
}

QVariant VLinkListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) return {};
    
    auto* xf = xform();
    if (!xf || index.row() < 0 || index.row() >= static_cast<int>(xf->vchain.size()))
        return {};

    switch (role) {
    case IndexRole:
        return index.row();
    case HandleRole:
        return QVariant::fromValue(getOrCreateHandle(index.row()));
    default:
        return {};
    }
}

QHash<int, QByteArray> VLinkListModel::roleNames() const {
    return {
        {IndexRole, "index"},
        {HandleRole, "handle"}
    };
}

VLinkHandle* VLinkListModel::getOrCreateHandle(int index) const {
    if (index < 0) return nullptr;
    
    // Expand cache if needed
    if (static_cast<int>(handleCache_.size()) <= index) {
        handleCache_.resize(index + 1, nullptr);
    }
    
    if (!handleCache_[index]) {
        handleCache_[index] = new VLinkHandle(flame_, xformIndex_, index,
                                              const_cast<VLinkListModel*>(this));
    }
    
    return handleCache_[index];
}

void VLinkListModel::refresh() {
    beginResetModel();
    for (auto* handle : handleCache_) {
        delete handle;
    }
    handleCache_.clear();
    endResetModel();
}

VLinkHandle* VLinkListModel::get(int index) {
    return getOrCreateHandle(index);
}

// ============================================================================
// XformHandle
// ============================================================================

XformHandle::XformHandle(rfkt::flame* flame, int index, QObject* parent)
    : QObject(parent)
    , flame_(flame)
    , index_(index)
    , vlinksModel_(flame, index, this)
{
}

rfkt::xform* XformHandle::xform() const {
    if (!flame_) return nullptr;
    return flame_->get_xform(index_);
}

double XformHandle::weight() const {
    if (auto* xf = xform()) return xf->weight.t0;
    return 0.0;
}

double XformHandle::color() const {
    if (auto* xf = xform()) return xf->color.t0;
    return 0.0;
}

double XformHandle::colorSpeed() const {
    if (auto* xf = xform()) return xf->color_speed.t0;
    return 0.0;
}

double XformHandle::opacity() const {
    if (auto* xf = xform()) return xf->opacity.t0;
    return 1.0;
}

void XformHandle::setWeight(double w) {
    if (auto* xf = xform()) { xf->weight.t0 = w; emit weightChanged(); }
}

void XformHandle::setColor(double c) {
    if (auto* xf = xform()) { xf->color.t0 = c; emit colorChanged(); }
}

void XformHandle::setColorSpeed(double cs) {
    if (auto* xf = xform()) { xf->color_speed.t0 = cs; emit colorSpeedChanged(); }
}

void XformHandle::setOpacity(double o) {
    if (auto* xf = xform()) { xf->opacity.t0 = o; emit opacityChanged(); }
}

// ============================================================================
// XformListModel
// ============================================================================

XformListModel::XformListModel(rfkt::flame* flame, QObject* parent)
    : QAbstractListModel(parent)
    , flame_(flame)
{
}

void XformListModel::setFlame(rfkt::flame* flame) {
    beginResetModel();
    flame_ = flame;
    for (auto* handle : handleCache_) {
        delete handle;
    }
    handleCache_.clear();
    delete finalXformHandle_;
    finalXformHandle_ = nullptr;
    endResetModel();
}

int XformListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid() || !flame_) return 0;
    return static_cast<int>(flame_->xforms().size()) + (flame_->final_xform ? 1 : 0);
}

int XformListModel::xformIndexFromRow(int row) const {
    if (!flame_) return -2;
    int regularCount = static_cast<int>(flame_->xforms().size());
    if (row < regularCount) return row;
    if (row == regularCount && flame_->final_xform) return -1;  // final xform
    return -2;  // invalid
}

QVariant XformListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || !flame_) return {};
    
    int xformIdx = xformIndexFromRow(index.row());
    if (xformIdx == -2) return {};

    switch (role) {
    case IndexRole:
        return xformIdx;
    case IsFinalRole:
        return xformIdx == -1;
    case HandleRole:
        return QVariant::fromValue(getOrCreateHandle(xformIdx));
    default:
        return {};
    }
}

QHash<int, QByteArray> XformListModel::roleNames() const {
    return {
        {IndexRole, "index"},
        {IsFinalRole, "isFinal"},
        {HandleRole, "handle"}
    };
}

XformHandle* XformListModel::getOrCreateHandle(int index) const {
    if (index == -1) {
        // Final xform
        if (!finalXformHandle_) {
            finalXformHandle_ = new XformHandle(flame_, -1,
                                                const_cast<XformListModel*>(this));
        }
        return finalXformHandle_;
    }
    
    if (index < 0) return nullptr;
    
    // Regular xform - expand cache if needed
    if (static_cast<int>(handleCache_.size()) <= index) {
        handleCache_.resize(index + 1, nullptr);
    }
    
    if (!handleCache_[index]) {
        handleCache_[index] = new XformHandle(flame_, index,
                                              const_cast<XformListModel*>(this));
    }
    
    return handleCache_[index];
}

void XformListModel::refresh() {
    beginResetModel();
    for (auto* handle : handleCache_) {
        delete handle;
    }
    handleCache_.clear();
    delete finalXformHandle_;
    finalXformHandle_ = nullptr;
    endResetModel();
}

XformHandle* XformListModel::get(int index) {
    return getOrCreateHandle(index);
}

// ============================================================================
// FlameHandle
// ============================================================================

FlameHandle::FlameHandle(QObject* parent)
    : QObject(parent)
    , xformsModel_(nullptr, this)
{
}

FlameHandle::FlameHandle(rfkt::flame* flame, QObject* parent)
    : QObject(parent)
    , flame_(flame)
    , xformsModel_(flame, this)
{
}

void FlameHandle::setFlame(rfkt::flame* flame) {
    flame_ = flame;
    xformsModel_.setFlame(flame);
    
    emit nameChanged();
    emit cameraChanged();
    emit renderSettingsChanged();
    emit paletteChanged();
    emit structureChanged();
}

QString FlameHandle::name() const {
    if (!flame_) return {};
    return QString::fromStdString(flame_->name);
}

void FlameHandle::setName(const QString& name) {
    if (flame_) {
        flame_->name = name.toStdString();
        emit nameChanged();
    }
}

double FlameHandle::centerX() const {
    return flame_ ? flame_->center_x.t0 : 0.0;
}

double FlameHandle::centerY() const {
    return flame_ ? flame_->center_y.t0 : 0.0;
}

double FlameHandle::scale() const {
    return flame_ ? flame_->scale.t0 : 1.0;
}

double FlameHandle::rotate() const {
    return flame_ ? flame_->rotate.t0 : 0.0;
}

void FlameHandle::setCenterX(double v) {
    if (flame_) { flame_->center_x.t0 = v; emit cameraChanged(); }
}

void FlameHandle::setCenterY(double v) {
    if (flame_) { flame_->center_y.t0 = v; emit cameraChanged(); }
}

void FlameHandle::setScale(double v) {
    if (flame_) { flame_->scale.t0 = v; emit cameraChanged(); }
}

void FlameHandle::setRotate(double v) {
    if (flame_) { flame_->rotate.t0 = v; emit cameraChanged(); }
}

double FlameHandle::gamma() const {
    return flame_ ? flame_->gamma.t0 : 4.0;
}

double FlameHandle::brightness() const {
    return flame_ ? flame_->brightness.t0 : 4.0;
}

double FlameHandle::vibrancy() const {
    return flame_ ? flame_->vibrancy.t0 : 1.0;
}

double FlameHandle::highlightPower() const {
    return flame_ ? flame_->highlight_power.t0 : 1.0;
}

double FlameHandle::gammaThreshold() const {
    return flame_ ? flame_->gamma_threshold.t0 : 0.01;
}

void FlameHandle::setGamma(double v) {
    if (flame_) { flame_->gamma.t0 = v; emit renderSettingsChanged(); }
}

void FlameHandle::setBrightness(double v) {
    if (flame_) { flame_->brightness.t0 = v; emit renderSettingsChanged(); }
}

void FlameHandle::setVibrancy(double v) {
    if (flame_) { flame_->vibrancy.t0 = v; emit renderSettingsChanged(); }
}

void FlameHandle::setHighlightPower(double v) {
    if (flame_) { flame_->highlight_power.t0 = v; emit renderSettingsChanged(); }
}

void FlameHandle::setGammaThreshold(double v) {
    if (flame_) { flame_->gamma_threshold.t0 = v; emit renderSettingsChanged(); }
}

double FlameHandle::modHue() const {
    return flame_ ? flame_->mod_hue.t0 : 0.0;
}

double FlameHandle::modSat() const {
    return flame_ ? flame_->mod_sat.t0 : 0.0;
}

double FlameHandle::modVal() const {
    return flame_ ? flame_->mod_val.t0 : 0.0;
}

void FlameHandle::setModHue(double v) {
    if (flame_) { flame_->mod_hue.t0 = v; emit paletteChanged(); }
}

void FlameHandle::setModSat(double v) {
    if (flame_) { flame_->mod_sat.t0 = v; emit paletteChanged(); }
}

void FlameHandle::setModVal(double v) {
    if (flame_) { flame_->mod_val.t0 = v; emit paletteChanged(); }
}

bool FlameHandle::hasFinalXform() const {
    return flame_ && flame_->final_xform.has_value();
}

QVariant FlameHandle::getParameter(const QString& path) const {
    if (!flame_) return {};
    auto* anima = flame_->lookup(path.toStdString());
    if (!anima) return {};
    return anima->t0;
}

bool FlameHandle::setParameter(const QString& path, double value) {
    if (!flame_) return false;
    auto* anima = flame_->lookup(path.toStdString());
    if (!anima) return false;
    anima->t0 = value;
    return true;
}

  هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرياضية والروابط والتاجات والشفرة البرمجية دون ترجمة:

    .. _plotting_api:

    ================================
    التطوير باستخدام واجهة برمجة التطبيقات للرسم البياني
    ================================

    يحدد Scikit-learn واجهة برمجة تطبيقات بسيطة لإنشاء مرئيات للتعلم الآلي. تتمثل الميزات الأساسية لهذه الواجهة في تشغيل العمليات الحسابية مرة واحدة والحصول على المرونة لضبط المرئيات بعد الحقيقة. هذا القسم مخصص للمطورين الذين يرغبون في تطوير أدوات الرسم البياني أو صيانتها. بالنسبة للاستخدام، يجب على المستخدمين الرجوع إلى :ref:`دليل المستخدم <visualizations>`.

    نظرة عامة على واجهة برمجة التطبيقات للرسم البياني
    --------------------------------

    يتم تغليف هذا المنطق في كائن عرض حيث يتم تخزين البيانات المحسوبة ويتم إجراء الرسم البياني في طريقة `plot`. تحتوي طريقة `__init__` لكائن العرض فقط على البيانات اللازمة لإنشاء المرئيات. تأخذ طريقة `plot` محددات لها علاقة بالتصور فقط، مثل محاور matplotlib. ستقوم طريقة `plot` بتخزين فناني matplotlib كسمات مما يسمح بالتعديلات على النمط من خلال كائن العرض. يجب أن يحدد فئة `Display` طريقة أو طريقتين للفئة: `from_estimator` و `from_predictions`. تسمح هذه الطرق بإنشاء كائن `Display` من التقدير وبعض البيانات أو من القيم الحقيقية والمتوقعة. بعد إنشاء طرق الفئة هذه لكائن العرض مع القيم المحسوبة، ثم استدعاء طريقة العرض للكائن. لاحظ أن الطريقة `plot` تحدد السمات المتعلقة ب matplotlib، مثل الفنان للخط. هذا يسمح بالتخصيصات بعد استدعاء طريقة `plot`.

    على سبيل المثال، يحدد `RocCurveDisplay` الطرق والسمات التالية::

       class RocCurveDisplay:
           def __init__(self, fpr, tpr, roc_auc, estimator_name):
               ...
               self.fpr = fpr
               self.tpr = tpr
               self.roc_auc = roc_auc
               self.estimator_name = estimator_name

           @classmethod
           def from_estimator(cls, estimator, X, y):
               # الحصول على التوقعات
               y_pred = estimator.predict_proba(X)[:, 1]
               return cls.from_predictions(y, y_pred, estimator.__class__.__name__)

           @classmethod
           def from_predictions(cls, y, y_pred, estimator_name):
               # قم بحساب ROC من y و y_pred
               fpr, tpr, roc_auc = ...
               viz = RocCurveDisplay(fpr, tpr, roc_auc, estimator_name)
               return viz.plot()

           def plot(self, ax=None, name=None, **kwargs):
               ...
               self.line_ = ...
               self.ax_ = ax
               self.figure_ = ax.figure_

    اقرأ المزيد في :ref:`sphx_glr_auto_examples_miscellaneous_plot_roc_curve_visualization_api.py` ودليل المستخدم :ref:`<visualizations>`.

    الرسم البياني مع محاور متعددة
    ---------------------------

    تدعم بعض أدوات الرسم البياني مثل :func:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` و :class:`~sklearn.inspection.PartialDependenceDisplay` الرسم البياني على محاور متعددة. يتم دعم سيناريوهين مختلفين:

    1. إذا تم تمرير قائمة محاور، فإن `plot` سيتحقق مما إذا كان عدد المحاور يتوافق مع عدد المحاور التي يتوقعها ثم يرسم على تلك المحاور. 2. إذا تم تمرير محور واحد، فإن ذلك المحور يحدد مساحة لمحاور متعددة يتم وضعها. في هذه الحالة، نوصي باستخدام `~matplotlib.gridspec.GridSpecFromSubplotSpec` من matplotlib لتقسيم المساحة::

       import matplotlib.pyplot as plt
       from matplotlib.gridspec import GridSpecFromSubplotSpec

       fig, ax = plt.subplots()
       gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())

       ax_top_left = fig.add_subplot(gs[0, 0])
       ax_top_right = fig.add_subplot(gs[0, 1])
       ax_bottom = fig.add_subplot(gs[1, :])

    بشكل افتراضي، الكلمة `ax` في `plot` هي `None`. في هذه الحالة، يتم إنشاء المحور الفردي ويتم استخدام واجهة برمجة التطبيقات (API) الخاصة بـ Gridspec لإنشاء المناطق للرسم فيها.

    انظر على سبيل المثال، :meth:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` الذي يرسم خطوط متعددة ومخططات باستخدام هذه الواجهة. يتم حفظ المحاور التي تحدد مربع الحدود في سمة `bounding_ax_`. يتم تخزين المحاور الفردية التي تم إنشاؤها في ndarray `axes_`، والتي تتوافق مع موضع المحاور على الشبكة. يتم تعيين المواضع غير المستخدمة إلى `None`. علاوة على ذلك، يتم تخزين فناني matplotlib في `lines_` و `contours_` حيث يكون المفتاح هو الموضع الموجود على الشبكة. عندما يتم تمرير قائمة من المحاور، فإن `axes_`، `lines_`، و `contours_` تكون مصفوفة أحادية البعد تتوافق مع قائمة المحاور التي تم تمريرها.

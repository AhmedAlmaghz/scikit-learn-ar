==============
التصويرات
==============

يعرّف Scikit-learn واجهة بسيطة لإنشاء تصويرات للتعلم الآلي. الميزة الأساسية لهذه الواجهة هي السماح بإنشاء رسومات سريعة وتعديلات بصرية دون إعادة الحساب. نحن نوفر فئات `Display` التي تعرض طريقتين لإنشاء الرسومات: `from_estimator` و `from_predictions`. ستأخذ طريقة `from_estimator` مقدّرًا مجهزًا وبعض البيانات (`X` و `y`) وستنشئ كائن `Display`. في بعض الأحيان ، نرغب في حساب التنبؤات مرة واحدة فقط ويجب استخدام `from_predictions` بدلاً من ذلك. في المثال التالي ، نرسم منحنى ROC لآلة متجه الدعم المجهزة:

.. plot::
   :context: close-figs
   :align: center

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import RocCurveDisplay
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    y = y == 2  # make binary
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    svc = SVC(random_state=42)
    svc.fit(X_train, y_train)

    svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)

يعيد كائن `svc_disp` ويسمح لنا بمواصلة استخدام منحنى ROC الذي تم حسابه بالفعل ل SVC في الرسومات المستقبلية. في هذه الحالة ، يكون `svc_disp` هو: class:`~sklearn.metrics.RocCurveDisplay` الذي يخزن القيم المحسوبة كسمة تسمى `roc_auc` و `fpr` و `tpr`. كن على دراية بأنه يمكننا الحصول على التنبؤات من آلة متجه الدعم ثم استخدام `from_predictions` بدلاً من `from_estimator`. بعد ذلك ، نقوم بتدريب مُصنف الغابة العشوائية ونعيد رسم منحنى roc المحسوب مسبقًا باستخدام طريقة `plot` لكائن `Display`.

.. plot::
   :context: close-figs
   :align: center

    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(X_train, y_train)

    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
    svc_disp.plot(ax=ax, alpha=0.8)

لاحظ أننا نمرر `alpha=0.8` إلى وظائف الرسم لضبط قيم ألفا للمنحنيات.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_miscellaneous_plot_roc_curve_visualization_api.py`
* :ref:`sphx_glr_auto_examples_miscellaneous_plot_partial_dependence_visualization_api.py`
* :ref:`sphx_glr_auto_examples_miscellaneous_plot_display_object_visualization.py`
* :ref:`sphx_glr_auto_examples_calibration_plot_compare_calibration.py`

تتوفر أدوات الرسم
============================

كائنات العرض
---------------

.. currentmodule:: sklearn

.. autosummary::

   calibration.CalibrationDisplay
   inspection.PartialDependenceDisplay
   inspection.DecisionBoundaryDisplay
   metrics.ConfusionMatrixDisplay
   metrics.DetCurveDisplay
   metrics.PrecisionRecallDisplay
   metrics.PredictionErrorDisplay
   metrics.RocCurveDisplay
   model_selection.LearningCurveDisplay
   model_selection.ValidationCurveDisplay

    

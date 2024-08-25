.. _array_api:

================================
دعم واجهة برمجة التطبيقات الخاصة بالمصفوفات (تجريبي)
================================

.. currentmodule:: sklearn

تحديد مواصفات واجهة برمجة التطبيقات الخاصة بالمصفوفات (`Array API <https://data-apis.org/array-api/latest/>`_) معيارًا لبرمجة التطبيقات للتعامل مع المصفوفات لجميع مكتبات التلاعب بالمصفوفات التي تشبه NumPy. يتطلب دعم Scikit-learn لواجهة برمجة التطبيقات الخاصة بالمصفوفات تثبيت `array-api-compat <https://github.com/data-apis/array-api-compat>`__.

يمكن تكوين بعض مقدرات Scikit-learn التي تعتمد بشكل أساسي على NumPy (بدلاً من استخدام Cython) لتنفيذ المنطق الخوارزمي لأساليب `fit` و`predict` أو `transform` الخاصة بهم لقبول أي هياكل بيانات متوافقة مع واجهة برمجة التطبيقات للمصفوفات وتفويض العمليات تلقائيًا إلى مساحة الاسم الأساسية بدلاً من الاعتماد على NumPy.

في هذه المرحلة، يعتبر هذا الدعم **تجريبيًا** ويجب تمكينه صراحةً كما هو موضح في ما يلي.

.. note::
    حالياً، من المعروف أن `array-api-strict` و`cupy` و`PyTorch` فقط تعمل مع مقدرات Scikit-learn.

مثال الاستخدام
=============

فيما يلي مثال على مقتطف التعليمات البرمجية يوضح كيفية استخدام `CuPy <https://cupy.dev/>`_ لتشغيل :class:`~discriminant_analysis.LinearDiscriminantAnalysis` علي GPU::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn import config_context
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> import cupy

    >>> X_np, y_np = make_classification(random_state=0)
    >>> X_cu = cupy.asarray(X_np)
    >>> y_cu = cupy.asarray(y_np)
    >>> X_cu.device
    <CUDA Device 0>

    >>> with config_context(array_api_dispatch=True):
    ...     lda = LinearDiscriminantAnalysis()
    ...     X_trans = lda.fit_transform(X_cu, y_cu)
    >>> X_trans.device
    <CUDA Device 0>

بعد تدريب النموذج، ستكون السمات المجهزة التي هي صفائف أيضًا من نفس مساحة اسم Array API مثل بيانات التدريب. على سبيل المثال، إذا تم استخدام مساحة اسم Array API لـ CuPy للتدريب، فستكون السمات المجهزة على وحدة معالجة الرسومات. نحن نقدم أداة مساعدة تجريبية `_estimator_with_converted_arrays` تنقل سمات المقدر من Array API إلى صفيف ndarray::

    >>> from sklearn.utils._array_api import _estimator_with_converted_arrays
    >>> cupy_to_ndarray = lambda array : array.get()
    >>> lda_np = _estimator_with_converted_arrays(lda, cupy_to_ndarray)
    >>> X_trans = lda_np.transform(X_np)
    >>> type(X_trans)
    <class 'numpy.ndarray'>

دعم PyTorch
-----------

يتم دعم PyTorch Tensors عن طريق تعيين `array_api_dispatch=True` وتمرير التنسورات مباشرة::

    >>> import torch
    >>> X_torch = torch.asarray(X_np, device="cuda", dtype=torch.float32)
    >>> y_torch = torch.asarray(y_np, device="cuda", dtype=torch.float32)

    >>> with config_context(array_api_dispatch=True):
    ...     lda = LinearDiscriminantAnalysis()
    ...     X_trans = lda.fit_transform(X_torch, y_torch)
    >>> type(X_trans)
    <class 'torch.Tensor'>
    >>> X_trans.device.type
    'cuda'

.. _array_api_supported:

الدعم لواجهة برمجة التطبيقات الخاصة بالمصفوفات
================================

المقدرون والأدوات الأخرى في scikit-learn التي تدعم Array API متوافقة المدخلات.

المقدرون
--------

- :class:`decomposition.PCA` (مع `svd_solver="full"`,
  `svd_solver="randomized"` و`power_iteration_normalizer="QR"`)
- :class:`linear_model.Ridge` (مع `solver="svd"`)
- :class:`discriminant_analysis.LinearDiscriminantAnalysis` (مع `solver="svd"`)
- :class:`preprocessing.KernelCenterer`
- :class:`preprocessing.MaxAbsScaler`
- :class:`preprocessing.MinMaxScaler`
- :class:`preprocessing.Normalizer`

الميتا مقدرون
--------------

الميتا مقدرون الذين يقبلون Array API المدخلات بشرط أن المقدر الأساسي يفعل أيضًا:

- :class:`model_selection.GridSearchCV`
- :class:`model_selection.RandomizedSearchCV`
- :class:`model_selection.HalvingGridSearchCV`
- :class:`model_selection.HalvingRandomSearchCV`

المقاييس
--------

- :func:`sklearn.metrics.cluster.entropy`
- :func:`sklearn.metrics.accuracy_score`
- :func:`sklearn.metrics.d2_tweedie_score`
- :func:`sklearn.metrics.max_error`
- :func:`sklearn.metrics.mean_absolute_error`
- :func:`sklearn.metrics.mean_absolute_percentage_error`
- :func:`sklearn.metrics.mean_gamma_deviance`
- :func:`sklearn.metrics.mean_poisson_deviance` (يتطلب `تمكين دعم واجهة برمجة التطبيقات للمصفوفات لـ SciPy <https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html#using-array-api-standard-support>`_)
- :func:`sklearn.metrics.mean_squared_error`
- :func:`sklearn.metrics.mean_tweedie_deviance`
- :func:`sklearn.metrics.pairwise.additive_chi2_kernel`
- :func:`sklearn.metrics.pairwise.chi2_kernel`
- :func:`sklearn.metrics.pairwise.cosine_similarity`
- :func:`sklearn.metrics.pairwise.cosine_distances`
- :func:`sklearn.metrics.pairwise.euclidean_distances` (انظر :ref:`device_support_for_float64`)
- :func:`sklearn.metrics.pairwise.paired_cosine_distances`
- :func:`sklearn.metrics.pairwise.paired_euclidean_distances`
- :func:`sklearn.metrics.pairwise.rbf_kernel` (انظر :ref:`device_support_for_float64`)
- :func:`sklearn.metrics.r2_score`
- :func:`sklearn.metrics.zero_one_loss`

من المتوقع أن ينمو التغطية بمرور الوقت. يرجى متابعة `المسألة المتداولة المخصصة على GitHub <https://github.com/scikit-learn/scikit-learn/issues/22352>`_ لتتبع التقدم.

نوع قيم الإرجاع والسمات المركبة
------------------------------------

عند استدعاء الوظائف أو الأساليب باستخدام Array API المتوافقة المدخلات، تكون الاتفاقية هي إرجاع قيم صفيف من نفس النوع من حاوية الصفيف والجهاز مثل بيانات الإدخال.

وبالمثل، عند تركيب مقدر مع Array API المتوافقة المدخلات، ستكون السمات المركبة صفائف من نفس المكتبة مثل الإدخال وتخزن على نفس الجهاز. طريقة `predict` و`transform` بعد ذلك تتوقع مدخلات من نفس مكتبة المصفوفة والجهاز مثل البيانات التي تم تمريرها إلى طريقة `fit`.

لاحظ مع ذلك أن دوال التقييم التي تعيد قيمًا قياسية تعيد قياسات Python (عادةً ما تكون مثيل `float`) بدلاً من قيمة صفيف قياسي.

فحوصات المقدر الشائعة

    

  لإضافة علامة `array_api_support` لمقدر (estimator) للإشارة إلى أنه يدعم واجهة برمجة تطبيقات المصفوفة (Array API)، يمكنك اتباع الخطوات التالية. سيؤدي هذا إلى تمكين فحوصات مخصصة كجزء من الاختبارات الشائعة للتحقق من أن نتائج المقدرات هي نفسها عند استخدام NumPy العادي ومدخلات واجهة برمجة تطبيقات المصفوفة.

    لتشغيل هذه الفحوصات، تحتاج إلى تثبيت `array_api_compat <https://github.com/data-apis/array-api-compat>`_ في بيئة الاختبار الخاصة بك. لتشغيل المجموعة الكاملة من الفحوصات، تحتاج إلى تثبيت كل من `PyTorch <https://pytorch.org/>`_ و `CuPy <https://cupy.dev/>`_ وأن يكون لديك وحدة معالجة الرسومات (GPU). سيتم تخطي الفحوصات التي لا يمكن تنفيذها أو التي بها تبعيات مفقودة تلقائيًا. لذلك من المهم تشغيل الاختبارات باستخدام علامة `-v` لمعرفة الفحوصات التي يتم تخطيها:

    .. prompt:: bash $

        pip install array-api-compat  # وتثبيت المكتبات الأخرى حسب الحاجة
        pytest -k "array_api" -v

    .. _mps_support:

    ملاحظة حول دعم جهاز MPS
    --------------------------

    على نظام macOS، يمكن لـ PyTorch استخدام Metal Performance Shaders (MPS) للوصول إلى مسرعات الأجهزة (مثل مكون GPU الداخلي لشرائح M1 أو M2). ومع ذلك، فإن دعم جهاز MPS لـ PyTorch غير مكتمل في وقت الكتابة. راجع مشكلة GitHub التالية للحصول على مزيد من التفاصيل:

    - https://github.com/pytorch/pytorch/issues/77764

    لتفعيل دعم MPS في PyTorch، قم بتعيين متغير البيئة `PYTORCH_ENABLE_MPS_FALLBACK=1` قبل تشغيل الاختبارات:

    .. prompt:: bash $

        PYTORCH_ENABLE_MPS_FALLBACK=1 pytest -k "array_api" -v

    في وقت الكتابة، يجب أن تمر جميع اختبارات scikit-learn، ومع ذلك، قد لا تكون سرعة الحساب أفضل بالضرورة من استخدام جهاز CPU.

    .. _device_support_for_float64:

    ملاحظة حول دعم الجهاز لنوع البيانات ``float64``
    --------------------------------------

    ستقوم عمليات معينة داخل scikit-learn تلقائيًا بتنفيذ العمليات على قيم الفاصلة العائمة بدقة `float64` لمنع تجاوزات السعة وضمان الدقة (على سبيل المثال، :func:`metrics.pairwise.euclidean_distances`). ومع ذلك، فإن مجموعات معينة من مساحات المصفوفات والأجهزة، مثل `PyTorch على MPS` (انظر :ref:`mps_support`) لا تدعم نوع البيانات `float64`. في هذه الحالات، ستعود scikit-learn إلى استخدام نوع البيانات `float32` بدلاً من ذلك. يمكن أن يؤدي هذا إلى سلوك مختلف (عادةً نتائج غير مستقرة عدديًا) مقارنة بعدم استخدام إرسال واجهة برمجة تطبيقات المصفوفة أو استخدام جهاز يدعم `float64`.

.. currentmodule:: sklearn.feature_selection

.. _feature_selection:

=================
اختيار الميزات
=================

يمكن استخدام الفئات في وحدة :mod:`sklearn.feature_selection` لإجراء اختيار الميزات/تقليل الأبعاد على مجموعات العينات، إما لتحسين دقة التقديرات أو لتعزيز أدائها على مجموعات البيانات عالية الأبعاد للغاية.

.. _variance_threshold:

إزالة الميزات ذات التباين المنخفض
=================================

:class:`VarianceThreshold` هو نهج أساسي بسيط لاختيار الميزات. يزيل جميع الميزات التي لا يفي تباينها ببعض العتبات. بشكل افتراضي، يزيل جميع ميزات التباين الصفري، أي الميزات التي لها نفس القيمة في جميع العينات.

على سبيل المثال، لنفترض أن لدينا مجموعة بيانات مع ميزات منطقية، ونريد إزالة جميع الميزات التي تكون قيمتها إما واحد أو صفر (تشغيل أو إيقاف) في أكثر من 80% من العينات. الميزات المنطقية هي متغيرات برنولي عشوائية، ويتم إعطاء تباين هذه المتغيرات بواسطة

.. math:: \mathrm{Var}[X] = p(1 - p)

لذلك يمكننا الاختيار باستخدام العتبة ``.8 * (1 - .8)``::

  >>> from sklearn.feature_selection import VarianceThreshold
  >>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
  >>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
  >>> sel.fit_transform(X)
 array([[0, 1],
         [1, 0],
         [0, 0],
         [1, 1],
         [1, 0],
         [1, 1]])

كما هو متوقع، أزال ``VarianceThreshold`` العمود الأول، والذي يحتوي على احتمال :math:`p = 5/6 > .8` لاحتواء صفر.

.. _univariate_feature_selection:

اختيار الميزات أحادية المتغير
============================

يعمل اختيار الميزات أحادية المتغير عن طريق اختيار أفضل الميزات بناءً على اختبارات إحصائية أحادية المتغير. يمكن اعتباره خطوة تمهيدية للمقدر. تعرض Scikit-learn إجراءات اختيار الميزات ككائنات تنفذ طريقة ``transform``:

* :class:`SelectKBest` يزيل جميع الميزات باستثناء :math:`k` الميزات الأعلى تسجيلًا

* :class:`SelectPercentile` يزيل جميع الميزات باستثناء نسبة مئوية محددة من قبل المستخدم من أعلى الميزات تسجيلًا

* باستخدام اختبارات إحصائية أحادية المتغير شائعة لكل ميزة: معدل الإيجابيات الكاذبة :class:`SelectFpr`، ومعدل الاكتشافات الكاذبة :class:`SelectFdr`، أو خطأ العائلة :class:`SelectFwe`.

* :class:`GenericUnivariateSelect` يسمح بإجراء اختيار الميزات أحادية المتغير باستراتيجية قابلة للتكوين. يتيح هذا اختيار أفضل استراتيجية اختيار أحادية المتغير مع البحث في تقدير المعلمات الزائدة.

على سبيل المثال، يمكننا استخدام اختبار F لاسترداد أفضل ميزتين لمجموعة بيانات كما يلي:

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectKBest
  >>> from sklearn.feature_selection import f_classif
  >>> X, y = load_iris(return_X_y=True)
  >>> X.shape
  (150, 4)
  >>> X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
  >>> X_new.shape
  (150, 2)

تأخذ هذه الكائنات كمدخلات دالة تسجيل تعيد درجات أحادية المتغير وقيم p (أو فقط الدرجات لـ :class:`SelectKBest` و :class:`SelectPercentile`):

* للتراجع: :func:`r_regression`, :func:`f_regression`, :func:`mutual_info_regression`

* للتصنيف: :func:`chi2`, :func:`f_classif`, :func:`mutual_info_classif`

تقدر طرق اختبار F درجة التبعية الخطية بين متغيرين عشوائيين. من ناحية أخرى، يمكن لطرق المعلومات المتبادلة التقاط أي نوع من التبعية الإحصائية، ولكن كونها لا معلمية، فإنها تتطلب المزيد من العينات للتقدير الدقيق. لاحظ أنه يجب تطبيق اختبار :math:`\chi^2` فقط على الميزات غير السلبية، مثل الترددات.

.. topic:: اختيار الميزات مع البيانات المتفرقة

   إذا كنت تستخدم بيانات متفرقة (أي بيانات ممثلة كمصفوفات متفرقة)، فإن :func:`chi2`, :func:`mutual_info_regression`, :func:`mutual_info_classif` ستتعامل مع البيانات دون جعلها كثيفة.

.. warning::

    احذر من عدم استخدام دالة تسجيل للتراجع مع مشكلة تصنيف، ستحصل على نتائج عديمة الفائدة.

.. note::

    يدعم :class:`SelectPercentile` و :class:`SelectKBest` أيضًا اختيار الميزات دون إشراف. تحتاج إلى توفير `score_func` حيث `y=None`. يجب أن تستخدم `score_func` داخليًا `X` لحساب الدرجات.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection_plot_feature_selection.py`

* :ref:`sphx_glr_auto_examples_feature_selection_plot_f_test_vs_mi.py`

.. _rfe:

إزالة الميزات بطريقة تكرارية
=============================

بالنظر إلى أداة تقييم خارجية تقوم بتعيين أوزان للميزات (مثل معاملات نموذج خطي)، فإن هدف الاستبعاد المتكرر للميزات (:class:`RFE`) هو اختيار الميزات من خلال النظر بشكل متكرر في مجموعات أصغر وأصغر من الميزات. أولاً، يتم تدريب الأداة على المجموعة الأولية من الميزات ويتم الحصول على أهمية كل ميزة إما من خلال أي سمة محددة (مثل ``coef_``، ``feature_importances_``) أو دوال قابلة للاستدعاء. بعد ذلك، يتم حذف أقل الميزات أهمية من المجموعة الحالية من الميزات. يتم تكرار هذا الإجراء بشكل متكرر على المجموعة المقصوصة حتى يتم الوصول إلى العدد المطلوب من الميزات للاختيار.

تقوم :class:`RFECV` بتنفيذ RFE في حلقة التحقق المتقاطع لإيجاد العدد الأمثل من الميزات. بتفاصيل أكثر، يتم ضبط عدد الميزات المحدد تلقائيًا عن طريق تركيب محدد :class:`RFE` على الانقسامات المختلفة للتحقق المتقاطع (المقدمة بواسطة معامل `cv`). يتم تقييم أداء محدد :class:`RFE` باستخدام `scorer` لأعداد مختلفة من الميزات المحددة ويتم تجميعها معًا. أخيرًا، يتم حساب متوسط الدرجات عبر الطيات ويتم تعيين عدد الميزات المحددة إلى عدد الميزات التي تزيد من درجة التحقق المتقاطع.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_digits.py`: مثال على استبعاد الميزات المتكرر يُظهر أهمية البكسلات في مهمة تصنيف الأرقام.

* :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`: مثال على استبعاد الميزات المتكرر مع الضبط التلقائي لعدد الميزات المحددة بالتحقق المتقاطع.

.. _select_from_model:

اختيار الميزة باستخدام SelectFromModel

    

:class:`SelectFromModel` هو محولٌ أعلى (meta-transformer) يمكن استخدامه إلى جانب أي مُقدر (estimator) يقوم بتعيين أهمية لكل ميزة من خلال سمة معينة (مثل `coef_`، `feature_importances_`) أو عبر استدعاء `importance_getter` بعد التخصيص. تعتبر الميزات غير مهمة ويتم إزالتها إذا كانت أهمية القيم المقابلة للميزة أقل من معلمّة `threshold` المقدمة. بالإضافة إلى تحديد قيمة حدّية رقمية، هناك إرشادات مدمجة لإيجاد قيمة حدّية باستخدام وسيط سلسلة. الإرشادات المتاحة هي "mean"، "median" وأعداد صحيحة مضاعفة لهذه القيم مثل "0.1*mean". وبالاشتراك مع معايير `threshold`، يمكن استخدام معلمة `max_features` لتحديد حد لعدد الميزات التي يجب اختيارها.

لمشاهدة أمثلة حول كيفية استخدامه، راجع الأقسام أدناه.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection_plot_select_from_model_diabetes.py`

.. _l1_feature_selection:

اختيار الميزات المستند إلى L1
-----------------------------

.. currentmodule:: sklearn

:ref:`الموديلات الخطية <linear_model>` المُعاقَبة بمعيار L1 لها حلولٌ متفرقة: العديد من معاملاتهم المقدرة تساوي الصفر. عندما يكون الهدف هو تقليل أبعاد البيانات لاستخدامها مع مصنفٍ آخر، يمكن استخدامها مع :class:`~feature_selection.SelectFromModel` لتحديد معاملات غير الصفر. على وجه الخصوص، المُقدرَات المتفرقة المفيدة لهذا الغرض هي :class:`~linear_model.Lasso` للتراجع (regression) و:class:`~linear_model.LogisticRegression` و:class:`~svm.LinearSVC` للتصنيف::

  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> X, y = load_iris(return_X_y=True)
  >>> X.shape
  (150, 4)
  >>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
  >>> model = SelectFromModel(lsvc, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape
  (150, 3)

مع SVMs والتراجع اللوجستي، تتحكم معلمة C في التفرق: كلما صغرت C قل عدد الميزات المحددة. مع Lasso، كلما كبرت قيمة المعلمة alpha، قل عدد الميزات المحددة.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_dense_vs_sparse_data.py`.

.. _compressive_sensing:

.. dropdown:: استعادة L1 والحساسية الضاغطة (compressive sensing)

  لاختيار جيد لمعلمة alpha، يمكن لـ :ref:`lasso` استعادة مجموعة الميزات غير الصفرية بالكامل باستخدام عدد قليل فقط من الملاحظات، بشرط استيفاء شروط محددة معينة. على وجه الخصوص، يجب أن يكون عدد العينات "كبيرًا بما فيه الكفاية"، وإلا ستعمل نماذج L1 بشكل عشوائي، حيث يعتمد "كبير بما فيه الكفاية" على عدد معاملات غير الصفر، لوغاريتم عدد الميزات، كمية الضوضاء، القيمة المطلقة الأصغر للمعاملات غير الصفرية، وبنية مصفوفة التصميم X. بالإضافة إلى ذلك، يجب أن تعرض مصفوفة التصميم خصائص معينة، مثل عدم وجود ترابط كبير.

  لا توجد قاعدة عامة لاختيار معلمة alpha لاستعادة معاملات غير الصفر. يمكن تعيينه بواسطة التحقق المتقاطع (:class:`~sklearn.linear_model.LassoCV` أو :class:`~sklearn.linear_model.LassoLarsCV`)، على الرغم من أن هذا قد يؤدي إلى نماذج ناقصة الجزاء: تضمين عدد صغير من المتغيرات غير ذات الصلة لا يضر بنتيجة التوقع. يميل BIC (:class:`~sklearn.linear_model.LassoLarsIC`) إلى تعيين قيم عالية من alpha.

  .. rubric:: المراجع

  Richard G. Baraniuk "Compressive Sensing", IEEE Signal
  Processing Magazine [120] July 2007
 http://users.isr.ist.utl.pt/~aguiar/CS_notes.pdf


اختيار الميزات المستند إلى الشجرة
---------------------------------

يمكن استخدام مقدرات الشجرة (راجع وحدة :mod:`sklearn.tree` والغابة الشجرية في وحدة :mod:`sklearn.ensemble`) لحساب أهمية الميزات المستند إلى النجاسة، والتي بدورها يمكن استخدامها للتخلص من الميزات غير ذات الصلة (عند اقترانها بمحولٍ أعلى :class:`~feature_selection.SelectFromModel`)::

  >>> from sklearn.ensemble import ExtraTreesClassifier
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> X, y = load_iris(return_X_y=True)
  >>> X.shape
  (150, 4)
  >>> clf = ExtraTreesClassifier(n_estimators=50)
  >>> clf = clf.fit(X, y)
  >>> clf.feature_importances_  # doctest: +SKIP
 array([ 0.04...,  0.05...,  0.4...,  0.4...])
  >>> model = SelectFromModel(clf, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape               # doctest: +SKIP
  (150, 2)

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances.py`: مثال على بيانات اصطناعية تظهر استعادة الميزات ذات المغزى الفعلي.

* :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances_faces.py`: مثال على بيانات التعرف على الوجه.

.. _sequential_feature_selection:

اختيار الميزة التسلسلي

    

تتوفر ميزة "اختيار الميزات التتابعية" (SFS) في المحول :class:`~sklearn.feature_selection.SequentialFeatureSelector`. يمكن أن يكون SFS إما أماميًا أو خلفيًا:

يعد SFS الأمامي إجراءً جشعًا يقوم بإيجاد أفضل ميزة جديدة لإضافتها إلى مجموعة الميزات المحددة بشكل تكراري. على وجه التحديد، نبدأ في البداية بدون ميزات ونبحث عن الميزة الواحدة التي تزيد من الدرجة المتقاطعة عندما يتم تدريب مقدر على هذه الميزة المفردة. بمجرد تحديد الميزة الأولى، نكرر الإجراء عن طريق إضافة ميزة جديدة إلى مجموعة الميزات المحددة. تتوقف العملية عندما يتم الوصول إلى العدد المطلوب من الميزات المحددة، كما يحدده معامل `n_features_to_select`.

يتبع SFS الخلفي الفكرة نفسها ولكنه يعمل في الاتجاه المعاكس: بدلاً من البدء بدون ميزات وإضافة الميزات بشكل جشع، نبدأ بجميع الميزات ونزيل الميزات بشكل جشع من المجموعة. يتحكم معامل `direction` فيما إذا كان يتم استخدام SFS الأمامي أو الخلفي.

.. dropdown:: تفاصيل حول اختيار الميزات التتابعية

  بشكل عام، لا يؤدي الاختيار الأمامي والخلفي إلى نتائج مكافئة. أيضًا، قد يكون أحدها أسرع بكثير من الآخر اعتمادًا على عدد الميزات المحددة المطلوبة: إذا كان لدينا 10 ميزات ونطلب 7 ميزات محددة، فستحتاج عملية الاختيار الأمامي إلى إجراء 7 تكرارات بينما تحتاج عملية الاختيار الخلفي إلى إجراء 3 تكرارات فقط.

  يختلف SFS عن :class:`~sklearn.feature_selection.RFE` و :class:`~sklearn.feature_selection.SelectFromModel` في أنه لا يحتاج إلى أن يعرض النموذج الأساسي سمة `coef_` أو `feature_importances_`. ومع ذلك، قد يكون أبطأ نظرًا لأنه يلزم تقييم المزيد من النماذج، مقارنةً بالطرق الأخرى. على سبيل المثال، في عملية الاختيار الخلفي، تتطلب التكرارات التي تنتقل من `m` ميزات إلى `m - 1` ميزات باستخدام التحقق المتقاطع مع k-fold تركيب `m * k` نماذج، بينما تتطلب :class:`~sklearn.feature_selection.RFE` تركيب واحد فقط، و :class:`~sklearn.feature_selection.SelectFromModel` دائمًا ما يفعل تركيب واحد فقط ولا يحتاج إلى تكرارات.

  .. rubric:: المراجع

  .. [sfs] Ferri et al, `دراسة مقارنة للتقنيات الخاصة باختيار الميزات على نطاق واسع <https://citeseerx.ist.psu.edu/doc_view/pid/5fedabbb3957bbb442802e012d829ee0629a01b6>`_.


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection_plot_select_from_model_diabetes.py`

اختيار الميزات كجزء من خط أنابيب
================================

عادةً ما يتم استخدام اختيار الميزات كخطوة مسبقة قبل إجراء التعلم الفعلي. الطريقة الموصى بها للقيام بذلك في scikit-learn هي استخدام :class:`~pipeline.Pipeline`::

 clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
    ('classification', RandomForestClassifier())
  ])
 clf.fit(X, y)

في هذا المقتطف، نستخدم :class:`~svm.LinearSVC`
المقترن مع :class:`~feature_selection.SelectFromModel`
لتقييم أهمية الميزات وتحديد الميزات الأكثر صلة.
بعد ذلك، يتم تدريب :class:`~ensemble.RandomForestClassifier` على المخرجات المحولة، أي باستخدام الميزات ذات الصلة فقط. يمكنك إجراء عمليات مماثلة باستخدام طرق اختيار الميزات الأخرى وأيضًا المصنفات التي توفر طريقة لتقييم أهمية الميزات بالطبع.
راجع أمثلة :class:`~pipeline.Pipeline` للحصول على مزيد من التفاصيل.

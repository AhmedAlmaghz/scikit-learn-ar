    .. _impute:

    ============================
    حذف القيم المفقودة
    ============================

    .. currentmodule:: sklearn.impute

    لأسباب مختلفة، تحتوي العديد من مجموعات البيانات الواقعية على قيم مفقودة، غالبًا ما يتم ترميزها كفراغات أو NaNs أو رموز نائبة أخرى. ومع ذلك، فإن هذه المجموعات من البيانات غير متوافقة مع تقديرات scikit-learn التي تفترض أن جميع القيم في المصفوفة رقمية، وأن جميعها لها معنى وتحمله. تتمثل الإستراتيجية الأساسية لاستخدام مجموعات البيانات غير الكاملة في تجاهل الصفوف و/أو الأعمدة بالكامل التي تحتوي على قيم مفقودة. ومع ذلك، يأتي هذا على حساب فقدان البيانات التي قد تكون قيمة (على الرغم من كونها غير مكتملة). تتمثل الإستراتيجية الأفضل في حساب القيم المفقودة، أي استنتاجها من الجزء المعروف من البيانات. انظر مدخل المصطلحات حول :term:`حذف القيم`.


    حذف القيم أحادية المتغير مقابل حذف القيم متعددة المتغيرات
    ======================================

    يوجد نوع من خوارزميات حذف القيم هو أحادي المتغير، والذي يحسب القيم في البعد i-th للميزة باستخدام القيم غير المفقودة فقط في هذا البعد للميزة (على سبيل المثال: :class:`SimpleImputer`). على النقيض من ذلك، تستخدم خوارزميات حذف القيم متعددة المتغيرات مجموعة كاملة من أبعاد الميزات المتاحة لتقدير القيم المفقودة (على سبيل المثال: :class:`IterativeImputer`).


    .. _single_imputer:

    حذف القيم أحادية المتغير
    =============================

    يوفر صنف :class:`SimpleImputer` استراتيجيات أساسية لحذف القيم المفقودة. يمكن حساب القيم المفقودة بقيمة ثابتة مقدمة، أو باستخدام إحصائيات (المتوسط أو الوسيط أو الأكثر تكرارًا) لكل عمود توجد فيه القيم المفقودة. يسمح هذا الصنف أيضًا بتشفير مختلف للقيم المفقودة.

    يوضح المقتطف التالي كيفية استبدال القيم المفقودة، المشفرة كـ ``np.nan``، باستخدام القيمة المتوسطة للأعمدة (المحور 0) التي تحتوي على القيم المفقودة::

        >>> import numpy as np
        >>> from sklearn.impute import SimpleImputer
        >>> imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        >>> imp.fit([[1, 2], [np.nan, 3], [7, 6]])
        SimpleImputer()
        >>> X = [[np.nan, 2], [6, np.nan], [7, 6]]
        >>> print(imp.transform(X))
        [[4.          2.        ]
         [6.          3.666...]
         [7.          6.        ]]

    يدعم صنف :class:`SimpleImputer` أيضًا المصفوفات المتفرقة::

        >>> import scipy.sparse as sp
        >>> X = sp.csc_matrix([[1, 2], [0, -1], [8, 4]])
        >>> imp = SimpleImputer(missing_values=-1, strategy='mean')
        >>> imp.fit(X)
        SimpleImputer(missing_values=-1)
        >>> X_test = sp.csc_matrix([[-1, 2], [6, -1], [7, 6]])
        >>> print(imp.transform(X_test).toarray())
        [[3. 2.]
         [6. 3.]
         [7. 6.]]

    لاحظ أن هذا التنسيق غير مخصص للاستخدام لتخزين القيم المفقودة ضمنيًا في المصفوفة لأنه سيؤدي إلى تكثيفها في وقت التحويل. يجب استخدام القيم المفقودة المشفرة بواسطة 0 مع الإدخال الكثيف.

    يدعم صنف :class:`SimpleImputer` أيضًا البيانات الفئوية الممثلة كقيم سلسلة أو فئات pandas عند استخدام إستراتيجية ``'most_frequent'`` أو ``'constant'``::

        >>> import pandas as pd
        >>> df = pd.DataFrame([["a", "x"],
        ...                    [np.nan, "y"],
        ...                    ["a", np.nan],
        ...                    ["b", "y"]], dtype="category")
        ...
        >>> imp = SimpleImputer(strategy="most_frequent")
        >>> print(imp.fit_transform(df))
        [['a' 'x']
         ['a' 'y']
         ['a' 'y']
         ['b' 'y']]

    لمزيد من الأمثلة حول الاستخدام، انظر :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.

    .. _iterative_imputer:


    حذف القيم متعددة المتغيرات

نهج أكثر تطوراً هو استخدام فئة :class:`IterativeImputer`، والتي تقوم نمذجة كل خاصية ذات قيم مفقودة كدالة للخصائص الأخرى، وتستخدم هذا التقدير للإحلال. تقوم بذلك بطريقة متكررة بالتناوب: في كل خطوة، يتم تعيين عمود خاصية كمخرج "y" ويتم التعامل مع أعمدة الخصائص الأخرى كمدخلات "X". يتم تركيب أداة تقويم على " (X، y) " لـ "y" المعروف. بعد ذلك، يتم استخدام أداة التقويم للتنبؤ بالقيم المفقودة لـ "y". يتم إجراء ذلك لكل خاصية بطريقة تكرارية، ثم يتم تكرارها لعدد "max_iter" من جولات الإحلال. يتم إرجاع نتائج جولة الإحلال النهائية.

.. note::

   هذا التقدير لا يزال **تجريبياً** في الوقت الحالي: قد تتغير المعلمات الافتراضية أو تفاصيل السلوك دون أي دورة إهمال. سيساعد حل المشكلات التالية في استقرار :class:`IterativeImputer`: معايير التقارب (:issue:`14338`)، والتقديرات الافتراضية (:issue:`13286`)، واستخدام الحالة العشوائية (:issue:`15611`). لاستخدامه، تحتاج إلى استيراد ``enable_iterative_imputer`` بشكل صريح.

::

    >>> import numpy as np
    >>> from sklearn.experimental import enable_iterative_imputer
    >>> from sklearn.impute import IterativeImputer
    >>> imp = IterativeImputer(max_iter=10, random_state=0)
    >>> imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])
    IterativeImputer(random_state=0)
    >>> X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
    >>> # يتعلم النموذج أن الخاصية الثانية هي ضعف الأولى
    >>> print(np.round(imp.transform(X_test)))
    [[ 1.  2.]
     [ 6. 12.]
     [ 3.  6.]]

يمكن استخدام كل من :class:`SimpleImputer` و :class:`IterativeImputer` في Pipeline كطريقة لبناء أداة تقدير مركبة تدعم الإحلال. انظر :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.

مرونة IterativeImputer
----------------------

هناك العديد من حزم الإحلال الراسخة في نظام R البيئي لعلوم البيانات: Amelia، mi، mice، missForest، إلخ. missForest شائع، ويتبين أنه حالة معينة من خوارزميات الإحلال التسلسلي المختلفة التي يمكن تنفيذها جميعًا باستخدام :class:`IterativeImputer` عن طريق تمرير أدوات تقويم مختلفة لاستخدامها للتنبؤ بالقيم المفقودة للخاصية. في حالة missForest، يكون أداة التقويم هذه هي Random Forest. انظر :ref:`sphx_glr_auto_examples_impute_plot_iterative_imputer_variants_comparison.py`.


.. _multiple_imputation:

الإحلال المتعدد مقابل الإحلال الفردي
-----------------------------------

في مجتمع الإحصاء، من الممارسات الشائعة إجراء عمليات إحلال متعددة، وتوليد، على سبيل المثال، "m" عمليات إحلال منفصلة لمصفوفة خاصية واحدة. ثم يتم وضع كل من عمليات الإحلال "m" هذه من خلال خط أنابيب التحليل اللاحق (على سبيل المثال، هندسة الميزات، التجميع، الانحدار، التصنيف). تسمح نتائج التحليل النهائية "m" (على سبيل المثال، أخطاء التحقق من الصحة المحتجزة) لعالم البيانات بالحصول على فهم لكيفية اختلاف نتائج التحليل كنتيجة لعدم اليقين المتأصل الناجم عن القيم المفقودة. تسمى الممارسة المذكورة أعلاه بالإحلال المتعدد.

كان تنفيذنا لـ :class:`IterativeImputer` مستوحى من حزمة R MICE (الإحلال متعدد المتغيرات بواسطة المعادلات المتسلسلة) [1]_، ولكنه يختلف عنها من خلال إرجاع إحلال واحد بدلاً من عمليات إحلال متعددة. ومع ذلك، يمكن أيضًا استخدام :class:`IterativeImputer` لعمليات إحلال متعددة عن طريق تطبيقه بشكل متكرر على نفس مجموعة البيانات ببذور عشوائية مختلفة عند ``sample_posterior=True``. انظر [2]_، الفصل 4 لمزيد من المناقشة حول الإحلال المتعدد مقابل الإحلال الفردي.

لا يزال من المشكلات المفتوحة مدى فائدة الإحلال الفردي مقابل الإحلال المتعدد في سياق التنبؤ والتصنيف عندما لا يكون المستخدم مهتمًا بقياس عدم اليقين بسبب القيم المفقودة.

لاحظ أنه لا يُسمح باستدعاء طريقة ``transform`` لـ :class:`IterativeImputer` بتغيير عدد العينات. لذلك لا يمكن تحقيق عمليات إحلال متعددة من خلال استدعاء واحد لـ ``transform``.

المراجع
----------

.. [1] `Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice: Multivariate
   Imputation by Chained Equations in R". Journal of Statistical Software 45:
   1-67. <https://www.jstatsoft.org/article/view/v045i03>`_

.. [2] Roderick J A Little and Donald B Rubin (1986). "Statistical Analysis
   with Missing Data". John Wiley & Sons, Inc., New York, NY, USA.

.. _knnimpute:

الإحلال باستخدام الجيران الأقرب
============================

توفر فئة :class:`KNNImputer` إمكانية ملء القيم المفقودة باستخدام نهج الجيران الأقرب k. بشكل افتراضي، يتم استخدام مقياس المسافة الإقليدية الذي يدعم القيم المفقودة، :func:`~sklearn.metrics.pairwise.nan_euclidean_distances`، لإيجاد الجيران الأقرب. يتم ملء كل خاصية مفقودة باستخدام القيم من ``n_neighbors`` من الجيران الأقرب الذين لديهم قيمة للخاصية. يمكن حساب متوسط الخاصية للجيران بشكل موحد أو مرجح بالمسافة لكل جار. إذا كان العينة لديها أكثر من خاصية مفقودة، فإن الجيران لتلك العينة يمكن أن يكونوا مختلفين اعتمادًا على الخاصية المحددة التي يتم ملؤها. عندما يكون عدد الجيران المتاحين أقل من `n_neighbors` ولا توجد مسافات محددة لمجموعة التدريب، يتم استخدام متوسط مجموعة التدريب لتلك الخاصية أثناء الملء. إذا كان هناك جار واحد على الأقل بمسافة محددة، فسيتم استخدام المتوسط المرجح أو غير المرجح للجيران المتبقين أثناء الملء. إذا كانت الخاصية مفقودة دائمًا في التدريب، فسيتم إزالتها أثناء `transform`. لمزيد من المعلومات حول المنهجية، راجع المرجع [OL2001]_.

يوضح المقتطف التالي كيفية استبدال القيم المفقودة، المشفرة كـ ``np.nan``، باستخدام متوسط قيمة الخاصية للجارين الأقرب للعينة ذات القيم المفقودة::

    >>> import numpy as np
    >>> from sklearn.impute import KNNImputer
    >>> nan = np.nan
    >>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
    >>> imputer = KNNImputer(n_neighbors=2, weights="uniform")
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
           [8. , 8. , 7. ]])

لمزيد من الأمثلة حول الاستخدام، راجع :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.

.. rubric:: المراجع

.. [OL2001] `Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor Hastie, Robert Tibshirani, David Botstein و Russ B. Altman، طرق تقدير القيم المفقودة لمصفوفات DNA، BIOINFORMATICS المجلد 17 رقم 6، 2001 الصفحات 520-525. <https://academic.oup.com/bioinformatics/article/17/6/520/272365>`_

الحفاظ على عدد الخصائص ثابتًا
=======================================

بشكل افتراضي، ستقوم أدوات ملء القيم المفقودة في scikit-learn بإسقاط الخصائص الفارغة تمامًا، أي الأعمدة التي تحتوي فقط على قيم مفقودة. على سبيل المثال::

  >>> imputer = SimpleImputer()
  >>> X = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 3]])
  >>> imputer.fit_transform(X)
 array([[1.],
         [2.],
         [3.]])

تم إسقاط الخاصية الأولى في `X` التي تحتوي فقط على `np.nan` بعد الملء. بينما لن تساعد هذه الخاصية في التنبؤ، فإن إسقاط الأعمدة سيغير شكل `X` مما قد يكون مشكلة عند استخدام أدوات ملء القيم المفقودة في خط أنابيب تعلم آلي أكثر تعقيدًا. يوفر المعلمة `keep_empty_features` خيار الاحتفاظ بالخصائص الفارغة عن طريق ملئها بقيم ثابتة. في معظم الحالات، تكون هذه القيمة الثابتة صفرًا::

  >>> imputer.set_params(keep_empty_features=True)
  SimpleImputer(keep_empty_features=True)
  >>> imputer.fit_transform(X)
 array([[0., 1.],
         [0., 2.],
         [0., 3.]])

.. _missing_indicator:

وضع علامة على القيم المملوءة
======================

إن محول :class:`MissingIndicator` مفيد لتحويل مجموعة بيانات إلى مصفوفة ثنائية مقابلة تشير إلى وجود قيم مفقودة في مجموعة البيانات. هذا التحويل مفيد بالاقتران مع الاستيفاء. عند استخدام الاستيفاء، يمكن أن يكون الحفاظ على المعلومات حول القيم المفقودة مفيدًا. لاحظ أن كل من :class:`SimpleImputer` و :class:`IterativeImputer` لهما معامل boolean `add_indicator` (الافتراضي هو `False`) والذي عند تعيينه على `True` يوفر طريقة ملائمة لتكديس مخرجات محول :class:`MissingIndicator` مع مخرجات المستفيد.

عادةً ما يتم استخدام `NaN` كحامل مكان للقيم المفقودة. ومع ذلك، فإنه يفرض نوع البيانات ليكون float. يسمح معامل `missing_values` بتحديد حامل مكان آخر مثل عدد صحيح. في المثال التالي، سنستخدم `-1` كقيم مفقودة::

  >>> from sklearn.impute import MissingIndicator
  >>> X = np.array([[-1, -1, 1, 3],
  ...               [4, -1, 0, -1],
  ...               [8, -1, 1, 0]])
  >>> indicator = MissingIndicator(missing_values=-1)
  >>> mask_missing_values_only = indicator.fit_transform(X)
  >>> mask_missing_values_only
 array([[ True,  True, False],
         [False,  True,  True],
         [False,  True, False]])

يتم استخدام معامل `features` لاختيار الميزات التي يتم بناء القناع لها. بشكل افتراضي، يكون `'missing-only'` الذي يعيد قناع المستفيد من الميزات التي تحتوي على قيم مفقودة في وقت `fit`::

  >>> indicator.features_
 array([0, 1, 3])

يمكن تعيين معامل `features` إلى `'all'` لإرجاع جميع الميزات سواء كانت تحتوي على قيم مفقودة أم لا::

  >>> indicator = MissingIndicator(missing_values=-1, features="all")
  >>> mask_all = indicator.fit_transform(X)
  >>> mask_all
 array([[ True,  True, False, False],
         [False,  True, False,  True],
         [False,  True, False, False]])
  >>> indicator.features_
 array([0, 1, 2, 3])

عند استخدام :class:`MissingIndicator` في :class:`~sklearn.pipeline.Pipeline`، تأكد من استخدام :class:`~sklearn.pipeline.FeatureUnion` أو :class:`~sklearn.compose.ColumnTransformer` لإضافة ميزات المؤشر إلى الميزات العادية. أولاً، نحصل على مجموعة بيانات `iris`، ونضيف بعض القيم المفقودة إليها.

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.impute import SimpleImputer, MissingIndicator
  >>> from sklearn.model_selection import train_test_split
  >>> from sklearn.pipeline import FeatureUnion, make_pipeline
  >>> from sklearn.tree import DecisionTreeClassifier
  >>> X, y = load_iris(return_X_y=True)
  >>> mask = np.random.randint(0, 2, size=X.shape).astype(bool)
  >>> X[mask] = np.nan
  >>> X_train, X_test, y_train, _ = train_test_split(X, y, test_size=100,
  ...                                                random_state=0)

الآن نقوم بإنشاء :class:`~sklearn.pipeline.FeatureUnion`. سيتم استيفاء جميع الميزات باستخدام :class:`SimpleImputer`، لتمكين التصنيفات من العمل مع هذه البيانات. بالإضافة إلى ذلك، فإنه يضيف متغيرات المؤشر من :class:`MissingIndicator`.

  >>> transformer = FeatureUnion(
  ...     transformer_list=[
  ...         ('features', SimpleImputer(strategy='mean')),
  ...         ('indicators', MissingIndicator())])
  >>> transformer = transformer.fit(X_train, y_train)
  >>> results = transformer.transform(X_test)
  >>> results.shape
  (100, 8)

بالطبع، لا يمكننا استخدام المحول لإجراء أي تنبؤات. يجب أن نلف هذا في :class:`~sklearn.pipeline.Pipeline` مع مصنف (على سبيل المثال، :class:`~sklearn.tree.DecisionTreeClassifier`) لتكون قادرة على إجراء تنبؤات.

  >>> clf = make_pipeline(transformer, DecisionTreeClassifier())
  >>> clf = clf.fit(X_train, y_train)
  >>> results = clf.predict(X_test)
  >>> results.shape
  (100,)

المقدرون المصممون للتعامل مع قيم NaN
=================================

تم تصميم بعض المقدرين للتعامل مع قيم NaN دون الحاجة إلى المعالجة المسبقة.
فيما يلي قائمة بهذه المقدرين، مصنفة حسب النوع (تجميع، رجعي، مصنف، تحويل):

.. allow_nan_estimators::
    

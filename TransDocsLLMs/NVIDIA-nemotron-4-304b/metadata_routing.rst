هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع مراعاة عدم ترجمة الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية:

.. currentmodule:: sklearn

.. TODO: update doc/conftest.py once document is updated and examples run.

.. _metadata_routing:

توجيه البيانات الوصفية (Metadata Routing)

(يرجى ملاحظة أن النص التالي هو ترجمة للتعليقات التوضيحية في النص الأصلي، ولا يتضمن ترجمة للرموز الخاصة أو الرموز والمعادلات الرياضية أو الروابط أو التاجات أو الشفرة البرمجية.)

  هذا هو النص الذي تم ترجمته إلى اللغة العربية:

    ================

    .. note::
      إن Metadata Routing API تجريبي، ولم يتم تنفيذه بعد لجميع التقديرات. يرجى الرجوع إلى :ref:`قائمة النماذج المدعومة وغير المدعومة <metadata_routing_models>` للحصول على مزيد من المعلومات. قد يتغير دون دورة إهمال معتادة. بشكل افتراضي، هذه الميزة غير مُمكنة. يمكنك تمكينها عن طريق تعيين علم ``enable_metadata_routing`` إلى ``True``::

        >>> import sklearn
        >>> sklearn.set_config(enable_metadata_routing=True)

      لاحظ أن الأساليب والمتطلبات المقدمة في هذه الوثيقة لا تكون ذات صلة إلا إذا كنت ترغب في تمرير :term:`metadata` (مثل ``sample_weight``) إلى طريقة ما. إذا كنت تقوم فقط بتمرير ``X`` و ``y`` ولا توجد أي معلمات / metadata أخرى للطرق مثل :term:`fit`, :term:`transform`, إلخ، فلا تحتاج إلى تعيين أي شيء.

    يوضح هذا الدليل كيفية :term:`metadata` يمكن توجيهها وتمريرها بين الكائنات في scikit-learn. إذا كنت تقوم بتطوير تقدير متوافق مع scikit-learn أو meta-estimator، فيمكنك التحقق من دليل المطور ذي الصلة: :ref:`sphx_glr_auto_examples_miscellaneous_plot_metadata_routing.py`.

    Metadata هي بيانات يأخذها التقدير أو المسجل أو مقسم السيرة الذاتية في الاعتبار إذا قام المستخدم بتمريرها صراحةً كمعامل. على سبيل المثال، :class:`~cluster.KMeans` تقبل `sample_weight` في طريقة `fit()` الخاصة به وتعتبره لحساب centroids الخاصة به. يتم استهلاك `classes` بواسطة بعض المصنفات ويتم استخدام `groups` في بعض الفواصل، ولكن يمكن اعتبار أي بيانات يتم تمريرها إلى طرق كائن بصرف النظر عن X و y على أنها metadata. قبل إصدار scikit-learn 1.3، لم يكن هناك واجهة برمجة تطبيقات واحدة لتمرير metadata مثل هذا إذا تم استخدام هذه الكائنات بالتزامن مع كائنات أخرى، على سبيل المثال مقسم يقبل `sample_weight` داخل :class:`~model_selection.GridSearchCV`.

    مع Metadata Routing API، يمكننا نقل metadata إلى التقديرات والمسجلين ومقسمي CV باستخدام :term:`meta-estimators` (مثل :class:`~pipeline.Pipeline` أو :class:`~model_selection.GridSearchCV`) أو وظائف مثل :func:`~model_selection.cross_validate` التي توجه البيانات إلى كائنات أخرى. من أجل تمرير metadata إلى طريقة مثل ``fit`` أو ``score``، يجب على الكائن الذي يستهلك metadata *طلب* ذلك. يتم ذلك عبر طرق `set_{method}_request()`, حيث يتم استبدال `{method}` باسم الطريقة التي تطلب metadata. على سبيل المثال، يستخدم المقدرون الذين يستخدمون metadata في طريقة `fit()` الخاصة بهم `set_fit_request()`, وسيستخدم المسجلون `set_score_request()`. تسمح هذه الأساليب بتحديد metadata التي سيتم طلبها، على سبيل المثال `set_fit_request(sample_weight=True)`.

    بالنسبة للفواصل المجمعة مثل :class:`~model_selection.GroupKFold`، يتم طلب معلم ``groups`` افتراضيًا. يوضح ذلك بشكل أفضل الأمثلة التالية.

    أمثلة الاستخدام
    **************
    فيما يلي بعض الأمثلة لإظهار بعض حالات الاستخدام الشائعة. هدفنا هو تمرير `sample_weight` و `groups` عبر :func:`~model_selection.cross_validate`, والذي يوجه metadata إلى :class:`~linear_model.LogisticRegressionCV` وإلى مسجل مخصص تم إنشاؤه باستخدام :func:`~metrics.make_scorer`, وكلاهما *يمكن* استخدام metadata في طرقهم. في هذه الأمثلة، نريد تعيين ما إذا كان سيتم استخدام metadata بشكل فردي ضمن :term:`consumers <consumer>` المختلفة.

    تتطلب الأمثلة في هذا القسم الواردات والبيانات التالية::

      >>> import numpy as np
      >>> from sklearn.metrics import make_scorer, accuracy_score
      >>> from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
      >>> from sklearn.model_selection import cross_validate, GridSearchCV, GroupKFold
      >>> from sklearn.feature_selection import SelectKBest
      >>> from sklearn.pipeline import make_pipeline
      >>> n_samples, n_features = 100, 4
      >>> rng = np.random.RandomState(42)
      >>> X = rng.rand(n_samples, n_features)
      >>> y = rng.randint(0, 2, size=n_samples)
      >>> my_groups = rng.randint(0, 10, size=n_samples)
      >>> my_weights = rng.rand(n_samples)
      >>> my_other_weights = rng.rand(n_samples)

    التسجيل والضبط المرجح
    ----------------------------

    يطلب مقسم :class:`~linear_model.LogisticRegressionCV` الداخلي, :class:`~model_selection.GroupKFold`, ``groups`` افتراضيًا. ومع ذلك، نحتاج إلى طلب `sample_weight` بشكل صريح بالنسبة له وللمسجل المخصص لدينا عن طريق تحديد `sample_weight=True` في طريقة `set_fit_request()` لـ :class:`~linear_model.LogisticRegressionCV` وفي طريقة `set_score_request()` لـ :func:`~metrics.make_scorer`. كلا :term:`consumers <consumer>` يعرفان كيفية استخدام ``sample_weight`` في طرق `fit()` أو `score()` الخاصة بهم. يمكننا بعد ذلك تمرير metadata في :func:`~model_selection.cross_validate` التي ستوجهها إلى أي مستهلكين نشطين::

      >>> weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
      >>> lr = LogisticRegressionCV(
      ...     cv=GroupKFold(),
      ...     scoring=weighted_acc
      ... ).set_fit_request(sample_weight=True)
      >>> cv_results = cross_validate(
      ...     lr,
      ...     X,
      ...     y,
      ...     params={"sample_weight": my_weights, "groups": my_groups},
      ...     cv=GroupKFold(),
      ...     scoring=weighted_acc,
      ... )

    لاحظ أنه في هذا المثال، توجه :func:`~model_selection.cross_validate` ``my_weights`` إلى كل من المسجل و:class:`~linear_model.LogisticRegressionCV`.

    إذا قمنا بتمرير `sample_weight` في معاملات :func:`~model_selection.cross_validate`, ولكن لم نضع أي كائن لطلبه، فسيتم رفع `UnsetMetadataPassedError`, مما يوحي بأننا بحاجة إلى تعيين المكان الذي سنوجهه إليه بشكل صريح. وينطبق الشيء نفسه إذا تم تمرير ``params={"sample_weights": my_weights, ...}`` (لاحظ الخطأ المطبعي، أي ``weights`` بدلاً من ``weight``), لأن ``sample_weights`` لم يتم طلبه من قبل أي من كائناته الأساسية.

    التسجيل المرجح والضبط غير المرجح

    (ملاحظة: لم يتم توفير المثال الأخير في النص الأصلي، لذا لن أقوم بترجمته.)
    
    (Note: The last example was not provided in the original text, so I will not translate it.)

عند تمرير بيانات وصفية مثل "sample_weight" إلى "موجّه" (:term:`meta-estimators` أو دالة توجيه)، تتطلب جميع "المستهلكين" :term:`consumers <consumer>` للأوزان أن يتم طلب الأوزان إما بشكل صريح أو بشكل صريح عدم طلبها (أي ``True`` أو ``False``). وبالتالي، لإجراء عملية ملائمة غير موزونة، نحتاج إلى تكوين :class:`~linear_model.LogisticRegressionCV` بحيث لا يطلب أوزان العينة، بحيث لا تقوم الدالة :func:`~model_selection.cross_validate` بتمرير الأوزان على طول::

  >>> weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
  >>> lr = LogisticRegressionCV(
  ...     cv=GroupKFold(), scoring=weighted_acc,
  ... ).set_fit_request(sample_weight=False)
  >>> cv_results = cross_validate(
  ...     lr,
  ...     X,
  ...     y,
  ...     cv=GroupKFold(),
  ...     params={"sample_weight": my_weights, "groups": my_groups},
  ...     scoring=weighted_acc,
  ... )

إذا لم يتم استدعاء :meth:`linear_model.LogisticRegressionCV.set_fit_request`، فإن الدالة :func:`~model_selection.cross_validate` ستُثير خطأ لأن ``sample_weight`` يتم تمريرها ولكن :class:`~linear_model.LogisticRegressionCV` لن يتم تكوينها بشكل صريح للتعرف على الأوزان.

إجراء اختيار الميزات غيرالموزونة
--------------------------------

لا يمكن تمرير بيانات وصفية إلا إذا كانت طريقة الكائن تعرف كيفية استخدام البيانات الوصفية، والتي في معظم الحالات تعني أن لديها معلمة صريحة. فقط عندها يمكننا تعيين قيم طلب للبيانات الوصفية باستخدام `set_fit_request(sample_weight=True)`، على سبيل المثال. هذا يجعل الكائن :term:`consumer <consumer>`.

على عكس :class:`~linear_model.LogisticRegressionCV`، لا يمكن لـ :class:`~feature_selection.SelectKBest` استهلاك الأوزان وبالتالي لا يتم تعيين قيمة طلب لـ ``sample_weight`` على مثيلها ولا يتم تمرير ``sample_weight`` إليها::

  >>> weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
  >>> lr = LogisticRegressionCV(
  ...     cv=GroupKFold(), scoring=weighted_acc,
  ... ).set_fit_request(sample_weight=True)
  >>> sel = SelectKBest(k=2)
  >>> pipe = make_pipeline(sel, lr)
  >>> cv_results = cross_validate(
  ...     pipe,
  ...     X,
  ...     y,
  ...     cv=GroupKFold(),
  ...     params={"sample_weight": my_weights, "groups": my_groups},
  ...     scoring=weighted_acc,
  ... )

أوزان التسجيل والملائمة المختلفة

ملاحظة: لم يتم ترجمة الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية.

هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية:

-------------------------------------

على الرغم من أن الدالتين :func:`~metrics.make_scorer` و :class:`~linear_model.LogisticRegressionCV` تتوقعان المفتاح ``sample_weight``، يمكننا استخدام الأسماء المستعارة لتمرير أوزان مختلفة إلى مستهلكين مختلفين. في هذا المثال، نمرر ``scoring_weight`` إلى الدالة المسجلة، و ``fitting_weight`` إلى :class:`~linear_model.LogisticRegressionCV`::

  >>> weighted_acc = make_scorer(accuracy_score).set_score_request(
  ...    sample_weight="scoring_weight"
  ... )
  >>> lr = LogisticRegressionCV(
  ...     cv=GroupKFold(), scoring=weighted_acc,
  ... ).set_fit_request(sample_weight="fitting_weight")
  >>> cv_results = cross_validate(
  ...     lr,
  ...     X,
  ...     y,
  ...     cv=GroupKFold(),
  ...     params={
  ...         "scoring_weight": my_weights,
  ...         "fitting_weight": my_other_weights,
  ...         "groups": my_groups,
  ...     },
  ...     scoring=weighted_acc,
  ... )

واجهة برمجة التطبيقات (API)
**************************

المستهلك (:term:`consumer`) هو كائن (تقدير، تقدير ميتا، مسجل، مقسم) الذي يقبل ويستخدم بعض البيانات الوصفية (:term:`metadata`) في طريقة واحدة على الأقل من طرقه (مثل ``fit``، ``predict``، ``inverse_transform``، ``transform``، ``score``، ``split``). التقديرات الفوقية (Meta-estimators) التي تقوم فقط بإعادة توجيه البيانات الوصفية إلى كائنات أخرى (تقديرات الطفل، المسجلات، أو المقسمات) ولا تستخدم البيانات الوصفية نفسها ليست مستهلكة. التقديرات (الميتا) التي توجّه البيانات الوصفية إلى كائنات أخرى هي :term:`routers <router>`. يمكن أن يكون التقدير (الميتا) :term:`consumer` و :term:`router` في نفس الوقت.

تعرض التقديرات (الميتا) والمقسمات طريقة `set_{method}_request` لكل طريقة تقبل بيانات وصفية واحدة على الأقل. على سبيل المثال، إذا كان التقدير يدعم ``sample_weight`` في ``fit`` و ``score``، فإنه يعرض ``estimator.set_fit_request(sample_weight=value)`` و ``estimator.set_score_request(sample_weight=value)``. هنا يمكن أن تكون قيمة ``value``:

- ``True``: تطلب الطريقة ``sample_weight``. هذا يعني أنه إذا تم توفير البيانات الوصفية، فسيتم استخدامها، وإلا فلن يتم رفع أي خطأ.
- ``False``: الطريقة لا تطلب ``sample_weight``.
- ``None``: سيرفع الموجّه خطأً إذا تم تمرير ``sample_weight``. هذا هو القيمة الافتراضية تقريبًا عند إنشاء كائن وضمان تعيين طلبات البيانات الوصفية بشكل صريح من قِبل المستخدم عند تمرير البيانات الوصفية. الاستثناء الوحيد هو مقسمات ``Group*Fold``.
- ``"param_name"``: اسم مستعار لـ ``sample_weight`` إذا أردنا تمرير أوزان مختلفة إلى مستهلكين مختلفين. إذا تم استخدام الاسم المستعار، فلا ينبغي أن يقوم التقدير الفوقي بإعادة توجيه ``"param_name"`` إلى المستهلك، ولكن ``sample_weight`` بدلاً من ذلك، لأن المستهلك سوف يتوقع وجود معلمة تسمى ``sample_weight``. هذا يعني أن التعيين بين البيانات الوصفية المطلوبة من قبل الكائن، على سبيل المثال ``sample_weight`` واسم المتغير الذي يوفره المستخدم، على سبيل المثال ``my_weights`` يتم على مستوى الموجّه، وليس بواسطة الكائن المستهلك نفسه.

يتم طلب البيانات الوصفية بنفس الطريقة للمسجلات باستخدام ``set_score_request``.

إذا تم تمرير البيانات الوصفية، على سبيل المثال ``sample_weight``، من قبل المستخدم، فيجب تعيين طلب البيانات الوصفية لجميع الكائنات التي يمكنها استهلاك ``sample_weight`` بشكل صريح من قبل المستخدم، وإلا سيرفع الكائن الموجّه خطأً. على سبيل المثال، يرفع الكود التالي خطأً، لأنه لم يتم تحديد ما إذا كان سيتم تمرير ``sample_weight`` إلى مسجل التقدير أم لا::

    >>> param_grid = {"C": [0.1, 1]}
    >>> lr = LogisticRegression().set_fit_request(sample_weight=True)
    >>> try:
    ...     GridSearchCV(
    ...         estimator=lr, param_grid=param_grid
    ...     ).fit(X, y, sample_weight=my_weights)
    ... except ValueError as e:
    ...     print(e)
    [sample_weight] are passed but are not explicitly set as requested or not
    requested for LogisticRegression.score, which is used within GridSearchCV.fit.
    Call `LogisticRegression.set_score_request({metadata}=True/False)` for each metadata
    you want to request/ignore.

يمكن حل المشكلة عن طريق تعيين قيمة الطلب بشكل صريح::

    >>> lr = LogisticRegression().set_fit_request(
    ...     sample_weight=True
    ... ).set_score_request(sample_weight=False)

في نهاية قسم **أمثلة الاستخدام**، نقوم بتعطيل علامة التهيئة لتوجيه البيانات الوصفية::

    >>> sklearn.set_config(enable_metadata_routing=False)

.. _metadata_routing_models:

حالة دعم توجيه البيانات الوصفية

    

جميع المستهلكين (أي المقدرات البسيطة التي تستهلك البيانات الوصفية فقط ولا توجّهها) يدعمون توجيه البيانات الوصفية، مما يعني أنه يمكن استخدامها داخل المقدرات الفوقية التي تدعم توجيه البيانات الوصفية. ومع ذلك، فإن تطوير الدعم لتوجيه البيانات الوصفية للمقدرات الفوقية قيد التقدم، وإليك قائمة بالمقدرات الفوقية والأدوات التي تدعم ولا تدعم توجيه البيانات الوصفية حتى الآن.

المقدرات الفوقية والوظائف التي تدعم توجيه البيانات الوصفية:

- :class:`sklearn.calibration.CalibratedClassifierCV`
- :class:`sklearn.compose.ColumnTransformer`
- :class:`sklearn.compose.TransformedTargetRegressor`
- :class:`sklearn.covariance.GraphicalLassoCV`
- :class:`sklearn.ensemble.StackingClassifier`
- :class:`sklearn.ensemble.StackingRegressor`
- :class:`sklearn.ensemble.VotingClassifier`
- :class:`sklearn.ensemble.VotingRegressor`
- :class:`sklearn.ensemble.BaggingClassifier`
- :class:`sklearn.ensemble.BaggingRegressor`
- :class:`sklearn.feature_selection.RFE`
- :class:`sklearn.feature_selection.RFECV`
- :class:`sklearn.feature_selection.SelectFromModel`
- :class:`sklearn.feature_selection.SequentialFeatureSelector`
- :class:`sklearn.impute.IterativeImputer`
- :class:`sklearn.linear_model.ElasticNetCV`
- :class:`sklearn.linear_model.LarsCV`
- :class:`sklearn.linear_model.LassoCV`
- :class:`sklearn.linear_model.LassoLarsCV`
- :class:`sklearn.linear_model.LogisticRegressionCV`
- :class:`sklearn.linear_model.MultiTaskElasticNetCV`
- :class:`sklearn.linear_model.MultiTaskLassoCV`
- :class:`sklearn.linear_model.OrthogonalMatchingPursuitCV`
- :class:`sklearn.linear_model.RANSACRegressor`
- :class:`sklearn.linear_model.RidgeClassifierCV`
- :class:`sklearn.linear_model.RidgeCV`
- :class:`sklearn.model_selection.GridSearchCV`
- :class:`sklearn.model_selection.HalvingGridSearchCV`
- :class:`sklearn.model_selection.HalvingRandomSearchCV`
- :class:`sklearn.model_selection.RandomizedSearchCV`
- :class:`sklearn.model_selection.permutation_test_score`
- :func:`sklearn.model_selection.cross_validate`
- :func:`sklearn.model_selection.cross_val_score`
- :func:`sklearn.model_selection.cross_val_predict`
- :class:`sklearn.model_selection.learning_curve`
- :class:`sklearn.model_selection.validation_curve`
- :class:`sklearn.multiclass.OneVsOneClassifier`
- :class:`sklearn.multiclass.OneVsRestClassifier`
- :class:`sklearn.multiclass.OutputCodeClassifier`
- :class:`sklearn.multioutput.ClassifierChain`
- :class:`sklearn.multioutput.MultiOutputClassifier`
- :class:`sklearn.multioutput.MultiOutputRegressor`
- :class:`sklearn.multioutput.RegressorChain`
- :class:`sklearn.pipeline.FeatureUnion`
- :class:`sklearn.pipeline.Pipeline`
- :class:`sklearn.semi_supervised.SelfTrainingClassifier`

المقدرات الفوقية والأدوات التي لا تدعم توجيه البيانات الوصفية حتى الآن:

- :class:`sklearn.ensemble.AdaBoostClassifier`
- :class:`sklearn.ensemble.AdaBoostRegressor`

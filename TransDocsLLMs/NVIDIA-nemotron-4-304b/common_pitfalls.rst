
.. _common_pitfalls:

=========================================
الأخطاء الشائعة والممارسات الموصى بها
=========================================

الغرض من هذا الفصل هو توضيح بعض الأخطاء الشائعة والأنماط غير الصحيحة التي تحدث عند استخدام scikit-learn. يوفر أمثلة على ما **لا** يجب فعله، إلى جانب مثال صحيح مقابِل.

التحضير المسبق غير المتسق
==========================

يوفر scikit-learn مكتبة من :ref:`data-transforms`، والتي قد تنظف (انظر :ref:`preprocessing`)، أو تقلل (انظر :ref:`data_reduction`)، أو توسع (انظر :ref:`kernel_approximation`)، أو تنشئ (انظر :ref:`feature_extraction`) تمثيلات للميزات. إذا تم استخدام هذه التحويلات للبيانات أثناء تدريب النموذج، فيجب استخدامها أيضًا في مجموعات البيانات اللاحقة، سواء كانت بيانات اختبار أو بيانات في نظام إنتاج. خلاف ذلك، سيتغير مساحة الميزة، ولن يكون النموذج قادرًا على الأداء بشكل فعال.

بالنسبة للمثال التالي، دعنا ننشئ مجموعة بيانات صناعية بميزة واحدة::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split

    >>> random_state = 42
    >>> X, y = make_regression(random_state=random_state, n_features=1, noise=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, random_state=random_state)

**خطأ**

تم تحجيم مجموعة بيانات التدريب، ولكن ليس مجموعة بيانات الاختبار، لذا فإن أداء النموذج على مجموعة بيانات الاختبار أسوأ من المتوقع::

    >>> from sklearn.metrics import mean_squared_error
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.preprocessing import StandardScaler

    >>> scaler = StandardScaler()
    >>> X_train_transformed = scaler.fit_transform(X_train)
    >>> model = LinearRegression().fit(X_train_transformed, y_train)
    >>> mean_squared_error(y_test, model.predict(X_test))
    62.80...

**صحيح**

بدلاً من تمرير `X_test` غير المحول إلى `predict`، يجب أن نحول بيانات الاختبار بنفس الطريقة التي حولنا بها بيانات التدريب::

    >>> X_test_transformed = scaler.transform(X_test)
    >>> mean_squared_error(y_test, model.predict(X_test_transformed))
    0.90...

بدلاً من ذلك، نوصي باستخدام :class:`Pipeline <sklearn.pipeline.Pipeline>`. مما يسهل ربط التحويلات مع المقدرين، ويقلل من احتمال نسيان التحويل::

    >>> from sklearn.pipeline import make_pipeline

    >>> model = make_pipeline(StandardScaler(), LinearRegression())
    >>> model.fit(X_train, y_train)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('linearregression', LinearRegression())])
    >>> mean_squared_error(y_test, model.predict(X_test))
    0.90...

تساعد Pipelines أيضًا في تجنب خطأ شائع آخر: تسريب بيانات الاختبار إلى بيانات التدريب.

.. _data_leakage:

تسريب البيانات

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
تحدث تسرب البيانات عندما يتم استخدام المعلومات التي لن تكون متاحة في وقت التنبؤ عند بناء النموذج. يؤدي هذا إلى تقديرات أداء متفائلة بشكل مفرط، على سبيل المثال من :ref:`التحقق المتقاطع <cross_validation>`, وبالتالي أداء أسوأ عند استخدام النموذج على بيانات جديدة بالفعل، على سبيل المثال أثناء الإنتاج.

سبب شائع هو عدم الحفاظ على مجموعات بيانات الاختبار والتدريب منفصلة. يجب عدم استخدام بيانات الاختبار مطلقًا لاتخاذ خيارات حول النموذج. **القاعدة العامة هي عدم استدعاء** `fit` **على بيانات الاختبار**. على الرغم من أن هذا قد يبدو واضحًا، إلا أنه من السهل تفويته في بعض الحالات، على سبيل المثال عند تطبيق خطوات معينة قبل المعالجة.

على الرغم من أن كل من مجموعات بيانات التدريب والاختبار يجب أن تتلقى نفس تحويل المعالجة المسبقة (كما هو موضح في القسم السابق)، فمن المهم أن يتم تعلم هذه التحويلات فقط من بيانات التدريب. على سبيل المثال، إذا كان لديك خطوة تطبيع حيث تقسم على القيمة المتوسطة، فيجب أن تكون القيمة المتوسطة هي متوسط مجموعة التدريب، **وليست** متوسط جميع البيانات. إذا تم تضمين مجموعة الاختبار في حساب المتوسط، فإن المعلومات من مجموعة الاختبار تؤثر على النموذج.

كيفية تجنب تسرب البيانات
-------------------------

فيما يلي بعض النصائح لتجنب تسرب البيانات:

* دائمًا قم بتقسيم البيانات إلى مجموعات فرعية للتدريب والاختبار أولاً، خاصةً قبل أي خطوات للمعالجة المسبقة.
* لا تقم مطلقًا بتضمين بيانات الاختبار عند استخدام طريقتي `fit` و `fit_transform`. يمكن أن يؤدي استخدام جميع البيانات، مثل `fit(X)`, إلى نتائج مفرطة في التفاؤل

  على العكس من ذلك، يجب استخدام طريقة `transform` على كل من مجموعات التدريب والاختبار حيث يجب تطبيق نفس المعالجة المسبقة على جميع البيانات. يمكن تحقيق ذلك باستخدام `fit_transform` على مجموعة التدريب و `transform` على مجموعة الاختبار.
* يعد :ref:` خط أنابيب <pipeline>` في scikit-learn طريقة رائعة لمنع تسرب البيانات حيث يضمن تنفيذ الطريقة المناسبة على مجموعة البيانات الصحيحة. يعد خط الأنابيب مثاليًا للاستخدام في وظائف التحقق المتقاطع وتوليف المعلمات.

تم تفصيل مثال على تسرب البيانات أثناء المعالجة المسبقة أدناه.

تسرب البيانات أثناء المعالجة المسبقة

(ملاحظة: لا توجد ترجمة للرموز الخاصة، والرموز والمعادلات الرياضية، والروابط والتاجات، والشفرة البرمجية كما هو مطلوب في السؤال)

.. note::
    اخترنا هنا توضيح تسريب البيانات بخطوة اختيار الخصائص.
    ومع ذلك، فإن خطر التسريب هذا ذو صلة تقريبًا بجميع التحويلات
    في scikit-learn، بما في ذلك (على سبيل المثال لا الحصر)
    :class:`~sklearn.preprocessing.StandardScaler`,
    :class:`~sklearn.impute.SimpleImputer`, و
    :class:`~sklearn.decomposition.PCA`.

يتوفر عدد من وظائف :ref:`feature_selection` في scikit-learn.
يمكنها المساعدة في إزالة الخصائص غير ذات الصلة والزائدة والضوضائية وكذلك
تحسين وقت إنشاء النموذج الخاص بك وأدائه. كما هو الحال مع أي نوع آخر من
المعالجة المسبقة، يجب أن يستخدم اختيار الخصائص **فقط** بيانات التدريب.
تضمين بيانات الاختبار في اختيار الخصائص سيؤثر بشكل متفائل على نموذجك.

للتوضيح، سننشئ مشكلة التصنيف الثنائي هذه بـ
10,000 خاصية تم إنشاؤها عشوائيًا::

    >>> import numpy as np
    >>> n_samples, n_features, n_classes = 200, 10000, 2
    >>> rng = np.random.RandomState(42)
    >>> X = rng.standard_normal((n_samples, n_features))
    >>> y = rng.choice(n_classes, n_samples)

**خطأ**

يؤدي استخدام جميع البيانات لإجراء اختيار الخصائص إلى الحصول على درجة دقة
أعلى بكثير من الصدفة، على الرغم من أن أهدافنا عشوائية تمامًا.
هذه العشوائية تعني أن `X` و `y` مستقلان، لذلك نتوقع
أن تكون الدقة حوالي 0.5. ومع ذلك، نظرًا لأن خطوة اختيار الخصائص "ترى" بيانات الاختبار،
فإن النموذج يتمتع بميزة غير عادلة. في المثال غير الصحيح أدناه، نستخدم أولاً كل البيانات
لاختيار الخصائص ثم نقسم البيانات إلى مجموعات تدريب واختبار لتركيب النموذج.
النتيجة هي درجة دقة أعلى بكثير مما هو متوقع::

    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.metrics import accuracy_score

    >>> # معالجة مسبقة غير صحيحة: يتم تحويل البيانات بالكامل
    >>> X_selected = SelectKBest(k=25).fit_transform(X, y)

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X_selected, y, random_state=42)
    >>> gbc = GradientBoostingClassifier(random_state=1)
    >>> gbc.fit(X_train, y_train)
    GradientBoostingClassifier(random_state=1)

    >>> y_pred = gbc.predict(X_test)
    >>> accuracy_score(y_test, y_pred)
    0.76

**صحيح**

لمنع تسريب البيانات، من الأفضل تقسيم بياناتك إلى مجموعات تدريب واختبار **أولًا**.
يمكن بعد ذلك إجراء اختيار الخصائص باستخدام مجموعة بيانات التدريب فقط. لاحظ أنه كلما استخدمنا
`fit` أو `fit_transform`، فإننا نستخدم مجموعة بيانات التدريب فقط. النتيجة الآن هي ما كنا نتوقعه
للبيانات، قريبة من الصدفة::

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42)
    >>> select = SelectKBest(k=25)
    >>> X_train_selected = select.fit_transform(X_train, y_train)

    >>> gbc = GradientBoostingClassifier(random_state=1)
    >>> gbc.fit(X_train_selected, y_train)
    GradientBoostingClassifier(random_state=1)

    >>> X_test_selected = select.transform(X_test)
    >>> y_pred = gbc.predict(X_test_selected)
    >>> accuracy_score(y_test, y_pred)
    0.46

مرة أخرى، نوصي باستخدام :class:`~sklearn.pipeline.Pipeline` لسلسلة
معا اختيار الخصائص وتقديرات النموذج. يضمن خط الأنابيب
أن بيانات التدريب فقط هي التي يتم استخدامها عند إجراء `fit` ويتم استخدام بيانات الاختبار
فقط لحساب درجة الدقة::

    >>> from sklearn.pipeline import make_pipeline
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42)
    >>> pipeline = make_pipeline(SelectKBest(k=25),
    ...                          GradientBoostingClassifier(random_state=1))
    >>> pipeline.fit(X_train, y_train)
    Pipeline(steps=[('selectkbest', SelectKBest(k=25)),
                    ('gradientboostingclassifier',
                    GradientBoostingClassifier(random_state=1))])

    >>> y_pred = pipeline.predict(X_test)
    >>> accuracy_score(y_test, y_pred)
    0.46

يمكن أيضًا تغذية خط الأنابيب في وظيفة التحقق المتقاطع
مثل :func:`~sklearn.model_selection.cross_val_score`.
مرة أخرى، يضمن خط الأنابيب استخدام مجموعة البيانات الفرعية وطريقة التقدير الصحيحة أثناء التركيب والتنبؤ::

    >>> from sklearn.model_selection import cross_val_score
    >>> scores = cross_val_score(pipeline, X, y)
    >>> print(f"Mean accuracy: {scores.mean():.2f}+/-{scores.std():.2f}")
    Mean accuracy: 0.46+/-0.07


.. _randomness:

التحكم في العشوائية

    

هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على رموز خاصة ورموز ومعادلات رياضية وروابط وتاجات وشفرة برمجية كما هي:

======================

بعض كائنات سكيت-ليرن عشوائية بطبيعتها. هذه عادةً ما تكون أدوات تقدير (مثل: :class:`~sklearn.ensemble.RandomForestClassifier`) ومقسِّمات التحقق المتقاطع (مثل: :class:`~sklearn.model_selection.KFold`). يتم التحكم في عشوائية هذه الكائنات عبر معامل `random_state` الخاص بها، كما هو موضح في :term:`Glossary <random_state>`. يوسع هذا القسم الإدخال في المسرد، ويصف الممارسات الجيدة والمزالق الشائعة فيما يتعلق بهذا المعامل الدقيق.

.. note:: ملخص التوصية

    للحصول على متانة مثالية لنتائج التحقق المتقاطع (CV)، مرّر أمثلة `RandomState` عند إنشاء أدوات التقدير، أو اترك `random_state` كـ `None`. يعد تمرير الأعداد الصحيحة إلى مقسِمات التحقق المتقاطع عادةً الخيار الأكثر أمانًا ويفضل؛ قد يكون تمرير أمثلة `RandomState` إلى المقسّمات مفيدًا أحيانًا لتحقيق حالات استخدام محددة للغاية.
    بالنسبة لكل من أدوات التقدير والمقسّمات، يؤدي تمرير عدد صحيح مقابل تمرير مثال (أو `None`) إلى اختلافات دقيقة ولكن مهمة، خاصة بالنسبة لإجراءات التحقق المتقاطع. من المهم فهم هذه الاختلافات عند الإبلاغ عن النتائج.

    للحصول على نتائج قابلة للتكرار عبر مرات التنفيذ، قم بإزالة أي استخدام لعبارة `random_state=None`.

استخدام `None` أو أمثلة `RandomState`، والمكالمات المتكررة إلى `fit` و`split`
--------------------------------------------------------------------------------

يحدد معامل `random_state` ما إذا كانت المكالمات المتعددة إلى :term:`fit` (بالنسبة لأدوات التقدير) أو إلى :term:`split` (بالنسبة لمقسّمات التحقق المتقاطع) ستنتج نفس النتائج، وفقًا للقواعد التالية:

- إذا تم تمرير عدد صحيح، فإن استدعاء `fit` أو `split` عدة مرات يؤدي دائمًا إلى نفس النتائج.
- إذا تم تمرير `None` أو مثال `RandomState`: فإن `fit` و`split` ستعطي نتائج مختلفة في كل مرة يتم استدعاؤها، وتستكشف سلسلة المكالمات جميع مصادر العشوائية. `None` هي القيمة الافتراضية لجميع معاملات `random_state`.

نوضح هنا هذه القواعد لكل من أدوات التقدير ومقسّمات التحقق المتقاطع.

.. note::
    نظرًا لأن تمرير `random_state=None` يعادل تمرير المثال العام `RandomState` من `numpy` (`random_state=np.random.mtrand._rand`)، فلن نذكر `None` هنا بشكل صريح. كل ما ينطبق على الأمثلة ينطبق أيضًا على استخدام `None`.

أدوات التقدير
..........

يعني تمرير أمثلة أن استدعاء `fit` عدة مرات لن يعطي نفس النتائج، حتى إذا تم تقييم أداة التقدير على نفس البيانات وبنفس المعلمات الفرعية::

    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np

    >>> rng = np.random.RandomState(0)
    >>> X, y = make_classification(n_features=5, random_state=rng)
    >>> sgd = SGDClassifier(random_state=rng)

    >>> sgd.fit(X, y).coef_
    array([[ 8.85418642,  4.79084103, -3.13077794,  8.11915045, -0.56479934]])

    >>> sgd.fit(X, y).coef_
    array([[ 6.70814003,  5.25291366, -7.55212743,  5.18197458,  1.37845099]])

يمكننا أن نرى من المقتطف أعلاه أن استدعاء `sgd.fit` بشكل متكرر قد أنتج نماذج مختلفة، حتى إذا كانت البيانات هي نفسها. وذلك لأن مولد الأرقام العشوائية (RNG) لأداة التقدير يتم استهلاكه (أي يتغير) عندما يتم استدعاء `fit`، وسيتم استخدام هذا RNG المتغير في المكالمات اللاحقة إلى `fit`. بالإضافة إلى ذلك، يتم مشاركة كائن `rng` عبر جميع الكائنات التي تستخدمه، وكنتيجة لذلك، تصبح هذه الكائنات إلى حد ما مترابطة. على سبيل المثال، سيؤثر اثنان من أدوات التقدير اللذان يشتركان في نفس مثال `RandomState` على بعضهما البعض، كما سنرى لاحقًا عند مناقشة الاستنساخ. هذه النقطة مهمة أن تضعها في الاعتبار عند التصحيح.

إذا كنا قد مررنا عددًا صحيحًا إلى معامل `random_state` لـ :class:`~sklearn.linear_model.SGDClassifier`، لكنا حصلنا على نفس النماذج، وبالتالي نفس النتائج في كل مرة. عندما نمرر عددًا صحيحًا، يتم استخدام نفس RNG عبر جميع المكالمات إلى `fit`. ما يحدث داخليًا هو أنه على الرغم من استهلاك RNG عند استدعاء `fit`، فإنه يتم دائمًا إعادة تعيينه إلى حالته الأصلية في بداية `fit`.

مقسّمات التحقق المتقاطع
............

تظهر مقسّمات التحقق المتقاطع العشوائية سلوكًا مشابهًا عند تمرير مثال `RandomState`؛ استدعاء `split` عدة مرات ينتج عنه انقسامات بيانات مختلفة::

    >>> from sklearn.model_selection import KFold
    >>> import numpy as np

    >>> X = y = np.arange(10)
    >>> rng = np.random.RandomState(0)
    >>> cv = KFold(n_splits=2, shuffle=True, random_state=rng)

    >>> for train, test in cv.split(X, y):
    ...     print(train, test)
    [0 3 5 6 7] [1 2 4 8 9]
    [1 2 4 8 9] [0 3 5 6 7]

    >>> for train, test in cv.split(X, y):
    ...     print(train, test)
    [0 4 6 7 8] [1 2 3 5 9]
    [1 2 3 5 9] [0 4 6 7 8]

يمكننا أن نرى أن الانقسامات مختلفة عن المرة الثانية التي يتم فيها استدعاء `split`. قد يؤدي هذا إلى نتائج غير متوقعة إذا قارنت أداء أدوات التقدير المتعددة عن طريق استدعاء `split` عدة مرات، كما سنرى في القسم التالي.

المزالق الشائعة والدقائق

    

  على الرغم من أن القواعد التي تحكم معيار "random_state" تبدو بسيطة، إلا أنها تترتب عليها بعض الآثار الدقيقة. في بعض الحالات، يمكن أن يؤدي ذلك حتى إلى استنتاجات خاطئة.

    المقدرون
    ..........

    **أنواع "random_state" المختلفة تؤدي إلى إجراءات مختلفة للتحقق المتقاطع**

    اعتمادًا على نوع معيار "random_state"، سيتصرف المقدّرون بشكل مختلف، خاصة في إجراءات التحقق المتقاطع. ضع في اعتبارك المقتطف التالي::

        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import cross_val_score
        >>> import numpy as np

        >>> X, y = make_classification(random_state=0)

        >>> rf_123 = RandomForestClassifier(random_state=123)
        >>> cross_val_score(rf_123, X, y)
        array([0.85, 0.95, 0.95, 0.9 , 0.9 ])

        >>> rf_inst = RandomForestClassifier(random_state=np.random.RandomState(0))
        >>> cross_val_score(rf_inst, X, y)
        array([0.9 , 0.95, 0.95, 0.9 , 0.9 ])

    نرى أن الدرجات المتقاطعة التحقق من "rf_123" و "rf_inst" مختلفة، كما هو متوقع نظرًا لأننا لم نمرر نفس معيار "random_state". ومع ذلك، فإن الفرق بين هذه الدرجات أمر أكثر دقة مما يبدو، و**إجراءات التحقق المتبادل التي أجرتها** :func:`~sklearn.model_selection.cross_val_score` **تختلف بشكل كبير في كل حالة**:

    - نظرًا لأن "rf_123" مرت بعدد صحيح، فإن كل استدعاء لـ "fit" يستخدم نفس مولد الأرقام العشوائية (RNG): هذا يعني أن جميع الخصائص العشوائية لمقدر الغابة العشوائية ستكون هي نفسها لكل من الأضعاف الخمسة لإجراء التحقق المتبادل. على وجه الخصوص، سيتم استخدام مجموعة (مختارة عشوائيًا) من ميزات المقدّر عبر جميع الأضعاف.
    - نظرًا لأن "rf_inst" مرت بمثيل "RandomState"، فإن كل استدعاء لـ "fit" يبدأ من RNG مختلف. نتيجة لذلك، ستكون مجموعة الميزات العشوائية مختلفة لكل طية.

    على الرغم من أن وجود RNG المقدّر الثابت عبر الأضعاف ليس خاطئًا بطبيعته، إلا أننا نريد عادةً نتائج CV قوية فيما يتعلق بعشوائية المقدّر. نتيجة لذلك، قد يكون من الأفضل تمرير مثيل بدلاً من عدد صحيح، لأنه سيسمح لمولد الأرقام العشوائية (RNG) للمقدر أن يختلف لكل طية.

    .. note::
        هنا، ستستخدم :func:`~sklearn.model_selection.cross_val_score` أداة تقسيم CV غير عشوائية (كما هو افتراضي)، لذلك سيتم تقييم كلا المقدرَين على نفس التقسيمات. هذه القسم ليس حول التباين في التقسيمات. أيضًا، سواء أكنّا نمرر عددًا صحيحًا أو مثيلًا إلى :func:`~sklearn.datasets.make_classification` أم لا، فهذا ليس ذا صلة بالغرض من التوضيح: ما يهم هو ما نمرره إلى مقدّر :class:`~sklearn.ensemble.RandomForestClassifier`.

    .. dropdown:: الاستنساخ

        يتمثل أحد الآثار الجانبية الدقيقة لتمرير مثيلات "RandomState" في كيفية عمل :func:`~sklearn.base.clone`::

            >>> from sklearn import clone
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> import numpy as np

            >>> rng = np.random.RandomState(0)
            >>> a = RandomForestClassifier(random_state=rng)
            >>> b = clone(a)

        نظرًا لأن مثيل "RandomState" مرر إلى "a"، فإن "a" و "b" ليسا مستنسختين بالمعنى الدقيق، ولكنهما نسخان في المعنى الإحصائي: "a" و "b" لا يزالان نموذجين مختلفين، حتى عند استدعاء "fit(X, y)" على نفس البيانات. علاوة على ذلك، سيتأثر "a" و "b" ببعضهما البعض نظرًا لأنهما يشتركان في نفس RNG الداخلي: استدعاء "a.fit" سيستهلك RNG لـ "b"، واستدعاء "b.fit" سيستهلك RNG لـ "a"، نظرًا لأنها هي نفسها. هذا الجزء صحيح بالنسبة لأي مقدرين يشتركان في معلمة "random_state"؛ إنه ليس خاصًا بالنسخ المستنسخة.

        إذا تم تمرير عدد صحيح، ستكون "a" و "b" نسخًا دقيقة ولن يؤثرا على بعضهما البعض.

        .. warning::
            على الرغم من أن :func:`~sklearn.base.clone` نادرًا ما يستخدم في رمز المستخدم، إلا أنه يتم استدعاؤه بشكل متكرر في جميع أنحاء قاعدة بيانات scikit-learn: على وجه الخصوص، فإن معظم المقدرّات الفوقية التي تقبل المقدّرين غير المثبتين تستدعي :func:`~sklearn.base.clone` داخليًا (:class:`~sklearn.model_selection.GridSearchCV`، :class:`~sklearn.ensemble.StackingClassifier`، :class:`~sklearn.calibration.CalibratedClassifierCV`، إلخ.).


    مقسمو CV
    ............

    عندما تمرر مثيل "RandomState"، فإن مقسمي CV يعطون تقسيمات مختلفة في كل مرة يتم استدعاء "split". عند مقارنة مقدّرين مختلفين، يمكن أن يؤدي هذا إلى المبالغة في تقدير تباين الفرق في الأداء بين المقدّرين::

        >>> from sklearn.naive_bayes import GaussianNB
        >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import KFold
        >>> from sklearn.model_selection import cross_val_score
        >>> import numpy as np

        >>> rng = np.random.RandomState(0)
        >>> X, y = make_classification(random_state=rng)
        >>> cv = KFold(shuffle=True, random_state=rng)
        >>> lda = LinearDiscriminantAnalysis()
        >>> nb = GaussianNB()

        >>> for est in (lda, nb):
        ...     print(cross_val_score(est, X, y, cv=cv))
        [0.8  0.75 0.75 0.7  0.85]
        [0.85 0.95 0.95 0.85 0.95]
    
    (ملاحظة: لقد حافظت على تنسيق الكود كما هو في النص الأصلي للتأكد من أن الترجمة دقيقة وسهلة القراءة.)

إن المقارنة المباشرة لأداء مقدر :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis` بمقدر :class:`~sklearn.naive_bayes.GaussianNB` **على كل طية** ستكون خطأ: **الانقسامات التي يتم تقييم المقدرين عليها مختلفة**. في الواقع، سوف تستدعي :func:`~sklearn.model_selection.cross_val_score` داخليًا `cv.split` على نفس مثيل :class:`~sklearn.model_selection.KFold`، ولكن الانقسامات ستكون مختلفة في كل مرة. هذا صحيح أيضًا بالنسبة لأي أداة تقوم باختيار النموذج عبر التحقق من الصحة المتقاطع، على سبيل المثال :class:`~sklearn.model_selection.GridSearchCV` و :class:`~sklearn.model_selection.RandomizedSearchCV`: لا يمكن مقارنة الدرجات من طية إلى طية عبر استدعاءات مختلفة لـ `search.fit`، نظرًا لأن `cv.split` سيكون قد تم استدعاؤه عدة مرات. ومع ذلك، داخل استدعاء واحد لـ `search.fit`، تكون المقارنة من طية إلى طية ممكنة نظرًا لأن مقدر البحث يستدعي `cv.split` مرة واحدة فقط.

للحصول على نتائج قابلة للمقارنة من طية إلى طية في جميع السيناريوهات، يجب المرور بعدد صحيح إلى أداة تقسيم CV: `cv = KFold(shuffle=True, random_state=0)`.

.. note::
    على الرغم من أن المقارنة من طية إلى طية غير مستحسنة مع مثيلات `RandomState`، إلا أنه يمكن توقع أن متوسط ​​الدرجات يسمح بالاستنتاج ما إذا كان أحد المقدرين أفضل من الآخر، بشرط استخدام طيات وبيانات كافية.

.. note::
    ما يهم في هذا المثال هو ما تم تمريره إلى :class:`~sklearn.model_selection.KFold`. سواء مررنا مثيل `RandomState` أو عدد صحيح إلى :func:`~sklearn.datasets.make_classification` فهو غير ذي صلة بغرض التوضيح لدينا. أيضًا، لا :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis` ولا :class:`~sklearn.naive_bayes.GaussianNB` مقدران عشوائيان.

توصيات عامة

    

(ملاحظة: تم الاحتفاظ بالرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية كما هي في النص الأصلي دون ترجمة.)

الحصول على نتائج قابلة للتكرار عبر تنفيذات متعددة
...............................................

للحصول على نتائج قابلة للتكرار (أي ثابتة) عبر تنفيذات متعددة لبرنامج، نحتاج إلى إزالة جميع استخدامات `random_state=None`، الذي هو القيمة الافتراضية. الطريقة الموصى بها هي إعلان متغير `rng` في أعلى البرنامج، وتمريره إلى أي كائن يقبل معلمة `random_state`::

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> import numpy as np

    >>> rng = np.random.RandomState(0)
    >>> X, y = make_classification(random_state=rng)
    >>> rf = RandomForestClassifier(random_state=rng)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=rng)
    >>> rf.fit(X_train, y_train).score(X_test, y_test)
    0.84

نحن الآن نضمن أن نتيجة هذا البرنامج ستكون دائمًا 0.84، بغض النظر عن عدد مرات تشغيله. سيؤثر تغيير متغير `rng` العالمي إلى قيمة مختلفة على النتائج، كما هو متوقع.

من الممكن أيضًا إعلان متغير `rng` كعدد صحيح. ومع ذلك، قد يؤدي هذا إلى نتائج أقل قوة للتحقق المتقاطع، كما سنرى في القسم التالي.

.. note::
    نحن لا نوصي بتعيين البذرة العالمية `numpy` عن طريق الاتصال `np.random.seed(0)`. راجع `هنا <https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array/5837352#comment6712034_583735

قوة نتائج التحقق المتقاطع
...........................

عندما نقيم أداء التقدير العشوائي عن طريق التحقق المتقاطع، نريد التأكد من أن التقدير يمكن أن يقدم توقعات دقيقة للبيانات الجديدة، لكننا نريد أيضًا التأكد من أن التقدير قوي فيما يتعلق بالتعلم العشوائي. على سبيل المثال، نريد أن تكون تهيئة الأوزان العشوائية لـ :class:`~sklearn.linear_model.SGDClassifier` جيدة باستمرار عبر جميع الطيات: وإلا، عندما ندرب ذلك التقدير على بيانات جديدة، قد نحصل على حظ سيئ وقد يؤدي التهيئة العشوائية إلى أداء سيئ. وبالمثل، نريد أن تكون الغابة العشوائية قوية فيما يتعلق بمجموعة الميزات المحددة عشوائيًا التي سيستخدمها كل شجرة.

لهذه الأسباب، يفضل تقييم أداء التحقق المتقاطع بالسماح للتقدير باستخدام RNG مختلف في كل طية. يتم ذلك عن طريق تمرير مثيل `RandomState` (أو `None`) إلى تهيئة التقدير.

عندما نمرر عددًا صحيحًا، سيستخدم التقدير نفس RNG في كل طية: إذا كان أداء التقدير جيدًا (أو سيئًا)، كما تم تقييمه بواسطة CV، فقد يكون ذلك مجرد لأننا حصلنا على الحظ (أو الحظ) مع تلك البذرة المحددة. يؤدي تمرير الحالات إلى نتائج أكثر قوة للتحقق المتقاطع، ويجعل المقارنة بين الخوارزميات المختلفة أكثر عدلاً. كما أنه يساعد في الحد من إغراء التعامل مع RNG التقدير كمعلمة فرط يمكن ضبطها.

سواء كنا نمرر مثيلات `RandomState` أو أعداد صحيحة إلى مقسمات CV، فلا يؤثر ذلك على قوة تحمل، طالما أن `split` تتم دعوتها مرة واحدة فقط. عندما تتم استدعاء `split` عدة مرات، لا يمكن مقارنة الطية بالطية بعد الآن. نتيجة لذلك، يعد تمرير الأعداد الصحيحة إلى مقسمات CV أكثر أمانًا ويغطي معظم حالات الاستخدام.


البدء

    (هذا النص بتنسيق RST، يرجى عدم ترجمة الرموز الخاصة، والرموز والمعادلات الرياضية، والروابط، والتاجات، والشفرة البرمجية)

الغرض من هذا الدليل هو توضيح بعض الميزات الرئيسية التي يوفرها "سكيكيت-ليرن" (scikit-learn). يفترض الدليل معرفة أساسية جدًا بممارسات التعلم الآلي (مثل تركيب النموذج، والتنبؤ، والتحقق المتقاطع، إلخ). يرجى الرجوع إلى :ref:`تعليمات التثبيت <installation-instructions>` لتثبيت "سكيكيت-ليرن".

"سكيكيت-ليرن" هي مكتبة تعلم آلي مفتوحة المصدر تدعم التعلم الخاضع للإشراف والتعلم غير الخاضع للإشراف. كما توفر أدوات مختلفة لتركيب النموذج، ومعالجة البيانات الأولية، واختيار النموذج، وتقييم النموذج، والعديد من الأدوات المساعدة الأخرى.

التركيب والتنبؤ: أساسيات التقدير
----------------------------------------

يوفر "سكيكيت-ليرن" عشرات من خوارزميات ونماذج التعلم الآلي المدمجة، والتي تسمى :term:`estimators`. يمكن تركيب كل أداة تقدير على بعض البيانات باستخدام طريقة :term:`fit`.

فيما يلي مثال بسيط حيث نركب :class:`~sklearn.ensemble.RandomForestClassifier` على بعض البيانات الأساسية جدًا::

  >>> from sklearn.ensemble import RandomForestClassifier
  >>> clf = RandomForestClassifier(random_state=0)
  >>> X = [[ 1,  2,  3],  # 2 samples, 3 features
  ...      [11, 12, 13]]
  >>> y = [0, 1]  # classes of each sample
  >>> clf.fit(X, y)
  RandomForestClassifier(random_state=0)

عادةً ما تقبل طريقة :term:`fit` مدخلين:

- مصفوفة العينات (أو مصفوفة التصميم) :term:`X`. حجم ``X`` عادةً ما يكون ``(n_samples, n_features)``، مما يعني أن العينات ممثلة كأسطر والميزات ممثلة كأعمدة.
- قيم الهدف :term:`y` التي تكون أرقامًا حقيقية لمهام الانحدار، أو أعدادًا صحيحة للتصنيف (أو أي مجموعة منفصلة أخرى من القيم). لمهام التعلم غير الخاضع للإشراف، لا يلزم تحديد ``y``. عادةً ما يكون ``y`` مجموعة أحادية البعد حيث يقابل الإدخال ``i`` الهدف للعيّنة ``i`` (الصف) من ``X``.

عادةً ما يُتوقع أن تكون كل من ``X`` و ``y`` مصفوفات نيمباي أو أنواع بيانات :term:`array-like` مكافئة، على الرغم من أن بعض أدوات التقدير تعمل مع تنسيقات أخرى مثل المصفوفات المتفرقة.

بمجرد تركيب أداة التقدير، يمكن استخدامها للتنبؤ بقيم الهدف للبيانات الجديدة. لا تحتاج إلى إعادة تدريب أداة التقدير::

  >>> clf.predict(X)  # predict classes of the training data
 array([0, 1])
  >>> clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data
 array([0, 1])

يمكنك التحقق من :ref:`ml_map` حول كيفية اختيار النموذج المناسب لحالتك.

المحولات ومعالجات ما قبل المعالجة
-------------------------------

تتكون عمليات تدفق العمل في التعلم الآلي غالبًا من أجزاء مختلفة. عادةً ما يتكون خط الأنابيب النموذجي من خطوة معالجة مسبقة تقوم بتحويل البيانات أو ملء الفراغات فيها، ونموذج تنبؤ نهائي يتوقع قيم الهدف.

في "سكيكيت-ليرن"، تتبع أدوات المعالجة المسبقة والمحولات نفس الواجهة البرمجية (API) ككائنات أداة التقدير (تستمد جميعها فعليًا من نفس الفئة ``BaseEstimator``). لا تحتوي كائنات المحول على طريقة :term:`predict` ولكنها تحتوي بدلاً من ذلك على طريقة :term:`transform` التي تنتج مصفوفة عينات جديدة ``X``::

  >>> from sklearn.preprocessing import StandardScaler
  >>> X = [[0, 15],
  ...      [1, -10]]
  >>> # scale data according to computed scaling values
  >>> StandardScaler().fit(X).transform(X)
 array([[-1.,  1.],
         [ 1., -1.]])

في بعض الأحيان، تريد تطبيق تحولات مختلفة على ميزات مختلفة: تم تصميم :ref:`ColumnTransformer<column_transformer>` لهذه الحالات.

خطوط الأنابيب: ربط معالجات ما قبل المعالجة وأدوات التقدير
--------------------------------------------------

يمكن دمج المحولات وأدوات التقدير (المنبئات) معًا في كائن موحد واحد: :class:`~sklearn.pipeline.Pipeline`. يوفر خط الأنابيب نفس الواجهة البرمجية (API) كأداة تقدير عادية: يمكن تركيبه واستخدامه للتنبؤ باستخدام ``fit`` و ``predict``. كما سنرى لاحقًا، سيمنعك استخدام خط الأنابيب أيضًا من تسريب البيانات، أي الكشف عن بعض بيانات الاختبار في بيانات التدريب الخاصة بك.

في المثال التالي، :ref:`نقوم بتحميل مجموعة بيانات Iris <datasets>`, ونقسمها إلى مجموعات تدريب واختبار، ونحسب درجة الدقة لنموذج خط أنابيب على بيانات الاختبار::

  >>> from sklearn.preprocessing import StandardScaler
  >>> from sklearn.linear_model import LogisticRegression
  >>> from sklearn.pipeline import make_pipeline
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.model_selection import train_test_split
  >>> from sklearn.metrics import accuracy_score
  ...
  >>> # create a pipeline object
  >>> pipe = make_pipeline(
  ...     StandardScaler(),
  ...     LogisticRegression()
  ... )
  ...
  >>> # load the iris dataset and split it into train and test sets
  >>> X, y = load_iris(return_X_y=True)
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  ...
  >>> # fit the whole pipeline
  >>> pipe.fit(X_train, y_train)
  Pipeline(steps=[('standardscaler', StandardScaler()),
                  ('logisticregression', LogisticRegression())])
  >>> # we can now use it like any other estimator
  >>> accuracy_score(pipe.predict(X_test), y_test)
  0.97...

تقييم النموذج
----

إن تركيب نموذج على بعض البيانات لا يعني بالضرورة أنه سيتوقع بشكل جيد على بيانات غير مرئية. يجب تقييم ذلك بشكل مباشر. لقد رأينا للتو مساعد :func:`~sklearn.model_selection.train_test_split` الذي يقسم مجموعة البيانات إلى مجموعات تدريب واختبار، ولكن ``scikit-learn`` يوفر العديد من الأدوات الأخرى لتقييم النموذج، وخاصة لـ :ref:`التحقق المتقاطع <cross_validation>`.

سنعرض هنا بإيجاز كيفية إجراء إجراء التحقق المتقاطع ذي 5 طيات، باستخدام مساعد :func:`~sklearn.model_selection.cross_validate`. لاحظ أنه من الممكن أيضًا التكرار يدويًا على الطيات، واستخدام استراتيجيات تقسيم بيانات مختلفة، واستخدام وظائف التهديف المخصصة. يرجى الرجوع إلى :ref:`دليل المستخدم <cross_validation>` لمزيد من التفاصيل::

  >>> from sklearn.datasets import make_regression
  >>> from sklearn.linear_model import LinearRegression
  >>> from sklearn.model_selection import cross_validate
  ...
  >>> X, y = make_regression(n_samples=1000, random_state=0)
  >>> lr = LinearRegression()
  ...
  >>> result = cross_validate(lr, X, y)  # defaults to 5-fold CV
  >>> result['test_score']  # r_squared score is high because dataset is easy
 array([1., 1., 1., 1., 1.])

  البحث التلقائي عن البارامتر
  ---------------------

  جميع المقيمين لديهم بارامترات (غالبًا ما تسمى بالفرط في الأدبيات) والتي يمكن ضبطها. غالبًا ما تعتمد قوة تعميم التقدير بشكل حاسم على بعض البارامتر. على سبيل المثال، يحتوي :class:`~sklearn.ensemble.RandomForestRegressor` على بارامتر ``n_estimators`` الذي يحدد عدد الأشجار في الغابة، وبارامتر ``max_depth`` الذي يحدد أقصى عمق لكل شجرة. في كثير من الأحيان، ليس من الواضح ما هي القيم الدقيقة لهذه البارامترات حيث أنها تعتمد على البيانات في متناول اليد.

  يوفر ``Scikit-learn`` أدوات للعثور تلقائيًا على أفضل مجموعات بارامتر (عبر التحقق المتقاطع). في المثال التالي، نقوم بالبحث عشوائيًا عبر مساحة بارامتر غابة عشوائية باستخدام كائن :class:`~sklearn.model_selection.RandomizedSearchCV`. عندما ينتهي البحث، يتصرف :class:`~sklearn.model_selection.RandomizedSearchCV` كـ :class:`~sklearn.ensemble.RandomForestRegressor` الذي تم تركيبه بأفضل مجموعة من البارامترات. اقرأ المزيد في :ref:`دليل المستخدم <grid_search>`::

  >>> from sklearn.datasets import fetch_california_housing
  >>> from sklearn.ensemble import RandomForestRegressor
  >>> from sklearn.model_selection import RandomizedSearchCV
  >>> from sklearn.model_selection import train_test_split
  >>> from scipy.stats import randint
  ...
  >>> X, y = fetch_california_housing(return_X_y=True)
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  ...
  >>> # define the parameter space that will be searched over
  >>> param_distributions = {'n_estimators': randint(1, 5),
  ...                        'max_depth': randint(5, 10)}
  ...
  >>> # now create a searchCV object and fit it to the data
  >>> search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
  ...                             n_iter=5,
  ...                             param_distributions=param_distributions,
  ...                             random_state=0)
  >>> search.fit(X_train, y_train)
  RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter=5,
                     param_distributions={'max_depth': ...,
                                          'n_estimators': ...},
                     random_state=0)
  >>> search.best_params_
  {'max_depth': 9, 'n_estimators': 4}

  >>> # the search object now acts like a normal random forest estimator
  >>> # with max_depth=9 and n_estimators=4
  >>> search.score(X_test, y_test)
  0.73...

.. note::

    في الممارسة، تريد دائمًا تقريبًا :ref:`البحث عبر خط أنابيب <composite_grid_search>`، بدلاً من مقيم واحد. أحد الأسباب الرئيسية هو أنه إذا طبقت خطوة ما قبل المعالجة على مجموعة البيانات بأكملها دون استخدام خط أنابيب، ثم أجريت أي نوع من التحقق المتقاطع، فستكون قد انتهكت الافتراض الأساسي للاستقلال بين بيانات التدريب والاختبار. في الواقع، نظرًا لأنك قمت بمعالجة البيانات باستخدام مجموعة البيانات بالكامل، فإن بعض المعلومات حول مجموعات الاختبار متاحة لمجموعات التدريب. سيؤدي هذا إلى المبالغة في تقدير قوة تعميم التقدير (يمكنك قراءة المزيد في هذا `مشاركة Kaggle <https://www.kaggle.com/alexisbcook/data-leakage>`_).

    سيبقيك استخدام خط الأنابيب للتحقق المتقاطع والبحث بعيدًا عن هذا الفخ الشائع إلى حد كبير.


الخطوات التالية
---------

لقد قمنا بتغطية تقدير التركيب والتوقع، خطوات ما قبل المعالجة، خطوط الأنابيب، أدوات التحقق المتقاطع وأدوات البحث التلقائي عن البارامتر. يجب أن يمنحك هذا الدليل نظرة عامة على بعض الميزات الرئيسية للمكتبة، ولكن هناك الكثير في ``scikit-learn``!

يرجى الرجوع إلى :ref:`دليل المستخدم <user_guide>` للحصول على تفاصيل حول جميع الأدوات التي نقدمها. يمكنك أيضًا العثور على قائمة شاملة بواجهة برمجة التطبيقات العامة في :ref:`api_ref`.

يمكنك أيضًا الاطلاع على العديد من :ref:`الأمثلة <general_examples>` التي توضح استخدام ``scikit-learn`` في العديد من السياقات المختلفة.

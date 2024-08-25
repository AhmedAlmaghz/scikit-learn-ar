
.. _combining_estimators:

==================================
الخطوط الأنبوبية والتقديرات المركبة
==================================

لبناء مُقدر مركب، عادةً ما يتم دمج المحولات مع محولات أخرى أو مع :term:`predictors` (مثل المصنفات أو معدلات الانحدار). الأداة الأكثر شيوعًا المستخدمة لتكوين المُقدرين هي :ref:`Pipeline <pipeline>`. تتطلب الخطوط الأنبوبية أن تكون جميع الخطوات باستثناء الأخيرة :term:`transformer`. يمكن أن تكون الخطوة الأخيرة أي شيء، محول، :term:`predictor`، أو مُقدر تجميع قد يكون له أو لا يكون لديه طريقة `.predict(...)`. تعرض Pipeline جميع الأساليب التي يوفرها المُقدر الأخير: إذا كانت الخطوة الأخيرة توفر طريقة `transform`، فستكون للخط الأنبوبي طريقة `transform` ويتصرف مثل المحول. إذا كانت الخطوة الأخيرة توفر طريقة `predict`، فإن Pipeline سيعرض هذه الطريقة، وبالنظر إلى بيانات :term:`X` ، يستخدم جميع الخطوات باستثناء الأخيرة لتحويل البيانات، ثم يعطي هذه البيانات المحولة إلى طريقة `predict` للخطوة الأخيرة من خط الأنابيب. غالبًا ما يتم استخدام الفصل :class:`Pipeline` جنبًا إلى جنب مع :ref:`ColumnTransformer <column_transformer>` أو :ref:`FeatureUnion <feature_union>` والتي تقوم بدمج مخرجات المحولات في مساحة ميزة مركبة. :ref:`TransformedTargetRegressor <transformed_target_regressor>` يتعامل مع تحويل :term:`target` (مثل تحويل السجل :term:`y`).

.. _pipeline:

الخط الأنبوبي: سلسلة من المُقدرين
=============================

.. currentmodule:: sklearn.pipeline

يمكن استخدام :class:`Pipeline` لسلسلة عدة مقدرات في واحد. هذا مفيد لأنه غالبًا ما يكون هناك تسلسل ثابت للخطوات في معالجة البيانات، مثل اختيار الميزات والتطبيع والتصنيف. تخدم :class:`Pipeline` عدة أغراض هنا:

الراحة والتغليف
    عليك فقط استدعاء :term:`fit` و :term:`predict` مرة واحدة على بياناتك لتناسب تسلسل كامل من المقدرات.
اختيار المعايير المشتركة
    يمكنك :ref:`grid search <grid_search>` على معايير جميع المقدرات في خط الأنابيب في وقت واحد.
الأمان
    تساعد خطوط الأنابيب في تجنب تسرب الإحصائيات من بيانات الاختبار الخاصة بك إلى النموذج المدرب في التحقق من الصحة المتقاطع، من خلال ضمان استخدام نفس العينات لتدريب المحولات والتنبؤات.

يجب أن تكون كل المقدرات في خط الأنابيب، باستثناء الأخير، محولات (أي يجب أن يكون لديها طريقة :term:`transform`).
قد يكون المقدّر الأخير من أي نوع (محول، مصنف، إلخ).

.. note::

    إن استدعاء ``fit`` على خط الأنابيب هو نفس استدعاء ``fit`` على كل مقدّر بدوره، ``transform`` إدخال البيانات وتمريرها إلى الخطوة التالية.
    يحتوي خط الأنابيب على جميع الطرق التي يمتلكها المقدّر الأخير في خط الأنابيب،
    على سبيل المثال، إذا كان المقدّر الأخير هو مصنف، فيمكن استخدام :class:`Pipeline` كمصنف. وإذا كان المقدّر الأخير هو محول، فإن خط الأنابيب هو أيضًا محول.

الاستخدام
    
    
   
إنشاء خط أنابيب
................

يتم إنشاء :class:`Pipeline` باستخدام قائمة من أزواج ``(key, value)``، حيث يكون ``key`` عبارة عن سلسلة تحتوي على الاسم الذي ترغب في إعطائه لهذه الخطوة و ``value`` عبارة عن كائن estimator::

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.svm import SVC
    >>> from sklearn.decomposition import PCA
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> pipe = Pipeline(estimators)
    >>> pipe
    Pipeline(steps=[('reduce_dim', PCA()), ('clf', SVC())])

.. dropdown:: نسخة مختصرة باستخدام :func:`make_pipeline`

  إن دالة :func:`make_pipeline` هي اختصار لبناء خطوط الأنابيب؛ فهي تأخذ عددًا متغيّرًا من estimators وتعيد خط أنابيب، وتملأ الأسماء تلقائيًا::

      >>> from sklearn.pipeline import make_pipeline
      >>> make_pipeline(PCA(), SVC())
      Pipeline(steps=[('pca', PCA()), ('svc', SVC())])

الوصول إلى خطوات خط الأنابيب
............................

يتم تخزين estimators لخط الأنابيب كقائمة في السمة ``steps``. ويمكن استخراج خط الأنابيب الفرعي باستخدام الترميز المقطعي الذي يستخدم عادةً للتسلسلات في Python مثل القوائم أو السلاسل (على الرغم من أنه لا يسمح إلا بخطوة واحدة):

    >>> pipe[:1]
    Pipeline(steps=[('reduce_dim', PCA())])
    >>> pipe[-1:]
    Pipeline(steps=[('clf', SVC())])

.. dropdown:: الوصول إلى خطوة باسم أو موضع

  يمكن أيضًا الوصول إلى خطوة محددة حسب الفهرس أو الاسم من خلال فهرسة (مع ``[idx]``) خط الأنابيب::

      >>> pipe.steps[0]
      ('reduce_dim', PCA())
      >>> pipe[0]
      PCA()
      >>> pipe['reduce_dim']
      PCA()

  تسمح سمة `Pipeline` المسماة `named_steps` بالوصول إلى الخطوات بالاسم مع الانتقال إلى علامة التبويب في البيئات التفاعلية::

      >>> pipe.named_steps.reduce_dim is pipe['reduce_dim']
      True

متابعة أسماء الخصائص في خط الأنابيب
.................................

لتمكين فحص النموذج، تحتوي :class:`~sklearn.pipeline.Pipeline` على طريقة ``get_feature_names_out()``، تمامًا مثل جميع المحولات. يمكنك استخدام تقطيع خط الأنابيب للحصول على أسماء الخصائص التي تدخل في كل خطوة::

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.feature_selection import SelectKBest
    >>> iris = load_iris()
    >>> pipe = Pipeline(steps=[
    ...    ('select', SelectKBest(k=2)),
    ...    ('clf', LogisticRegression())])
    >>> pipe.fit(iris.data, iris.target)
    Pipeline(steps=[('select', SelectKBest(...)), ('clf', LogisticRegression(...))])
    >>> pipe[:-1].get_feature_names_out()
    array(['x2', 'x3'], ...)

.. dropdown:: تخصيص أسماء الخصائص

  يمكنك أيضًا توفير أسماء خصائص مخصصة لبيانات الإدخال باستخدام ``get_feature_names_out``::

      >>> pipe[:-1].get_feature_names_out(iris.feature_names)
      array(['petal length (cm)', 'petal width (cm)'], ...)

.. _pipeline_nested_parameters:

الوصول إلى المعلمات المتداخلة
...........................

من الشائع ضبط معلمات estimator ضمن خط الأنابيب. هذه المعلمة متداخلة إذن لأنها تنتمي إلى خطوة فرعية معينة. يمكن الوصول إلى معلمات estimators في خط الأنابيب باستخدام بناء جملة ``<estimator>__<parameter>``::

    >>> pipe = Pipeline(steps=[("reduce_dim", PCA()), ("clf", SVC())])
    >>> pipe.set_params(clf__C=10)
    Pipeline(steps=[('reduce_dim', PCA()), ('clf', SVC(C=10))])

.. dropdown:: متى يكون هذا مهمًا؟

  هذا مهم بشكل خاص للقيام بعمليات بحث الشبكة::

      >>> from sklearn.model_selection import GridSearchCV
      >>> param_grid = dict(reduce_dim__n_components=[2, 5, 10],
      ...                   clf__C=[0.1, 10, 100])
      >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)

  ويمكن أيضًا استبدال الخطوات الفردية كمعلمات، ويمكن تجاهل الخطوات غير النهائية عن طريق تعيينها على ``'passthrough'``::

      >>> param_grid = dict(reduce_dim=['passthrough', PCA(5), PCA(10)],
      ...                   clf=[SVC(), LogisticRegression()],
      ...                   clf__C=[0.1, 10, 100])
      >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)

  .. seealso::

    * :ref:`composite_grid_search`


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection_plot_feature_selection_pipeline.py`
* :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_text_feature_extraction.py`
* :ref:`sphx_glr_auto_examples_compose_plot_digits_pipe.py`
* :ref:`sphx_glr_auto_examples_miscellaneous_plot_kernel_approximation.py`
* :ref:`sphx_glr_auto_examples_svm_plot_svm_anova.py`
* :ref:`sphx_glr_auto_examples_compose_plot_compare_reduction.py`
* :ref:`sphx_glr_auto_examples_miscellaneous_plot_pipeline_display.py`


.. _pipeline_cache:

المحولات المخفية: تجنب الحساب المتكرر
-------------------------------------------------

.. currentmodule:: sklearn.pipeline

قد يكون تركيب المحولات حسابياً مكلفاً. عند ضبط معامل ``memory``، ستقوم :class:`Pipeline` بتخزين كل محول مؤقتاً بعد استدعاء ``fit``.
تستخدم هذه الخاصية لتجنب حساب محولات التجهيز داخل خط الأنابيب إذا كانت المعاملات وبيانات الإدخال متطابقة. ومن الأمثلة النموذجية حالة البحث الشبكي التي يمكن فيها تركيب المحولات مرة واحدة وإعادة استخدامها لكل تشكيل. لن يتم تخزين الخطوة الأخيرة مؤقتاً، حتى إذا كانت محولاً.

يلزم وجود معامل ``memory`` لتخزين المحولات مؤقتاً. يمكن أن يكون ``memory`` إما سلسلة تحتوي على الدليل الذي سيتم فيه تخزين المحولات مؤقتاً أو كائن `joblib.Memory <https://joblib.readthedocs.io/en/latest/memory.html>`_::

    >>> from tempfile import mkdtemp
    >>> from shutil import rmtree
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.svm import SVC
    >>> from sklearn.pipeline import Pipeline
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> cachedir = mkdtemp()
    >>> pipe = Pipeline(estimators, memory=cachedir)
    >>> pipe
    Pipeline(memory=...,
             steps=[('reduce_dim', PCA()), ('clf', SVC())])
    >>> # Clear the cache directory when you don't need it anymore
    >>> rmtree(cachedir)

.. dropdown:: تأثير جانبي لتخزين المحولات مؤقتاً
  :color: warning

  عند استخدام :class:`Pipeline` بدون تمكين التخزين المؤقت، يمكن فحص المثيل الأصلي مثل::

      >>> from sklearn.datasets import load_digits
      >>> X_digits, y_digits = load_digits(return_X_y=True)
      >>> pca1 = PCA(n_components=10)
      >>> svm1 = SVC()
      >>> pipe = Pipeline([('reduce_dim', pca1), ('clf', svm1)])
      >>> pipe.fit(X_digits, y_digits)
      Pipeline(steps=[('reduce_dim', PCA(n_components=10)), ('clf', SVC())])
      >>> # The pca instance can be inspected directly
      >>> pca1.components_.shape
      (10, 64)

  يؤدي تمكين التخزين المؤقت إلى تشغيل استنساخ المحولات قبل التركيب. وبالتالي، لا يمكن فحص مثيل المحول المعطى لخط الأنابيب مباشرة.
  في المثال التالي، سيؤدي الوصول إلى مثيل :class:`~sklearn.decomposition.PCA` ``pca2`` إلى رفع ``AttributeError`` لأن ``pca2`` سيكون محول غير مجهز.
  بدلاً من ذلك، استخدم السمة ``named_steps`` لفحص أجهزة التقدير داخل خط الأنابيب::

      >>> cachedir = mkdtemp()
      >>> pca2 = PCA(n_components=10)
      >>> svm2 = SVC()
      >>> cached_pipe = Pipeline([('reduce_dim', pca2), ('clf', svm2)],
      ...                        memory=cachedir)
      >>> cached_pipe.fit(X_digits, y_digits)
      Pipeline(memory=...,
               steps=[('reduce_dim', PCA(n_components=10)), ('clf', SVC())])
      >>> cached_pipe.named_steps['reduce_dim'].components_.shape
      (10, 64)
      >>> # Remove the cache directory
      >>> rmtree(cachedir)


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_compose_plot_compare_reduction.py`

.. _transformed_target_regressor:

تحويل الهدف في الانحدار
  

يصف فئتين في مكتبة Scikit-learn لاستخدام التحويلات في التعلم الآلي: TransformedTargetRegressor و FeatureUnion.

TransformedTargetRegressor:

تقوم فئة TransformedTargetRegressor بتحويل المتغيرات الهدف "y" قبل تركيب نموذج الانحدار. يتم إعادة تعيين التوقعات إلى الفضاء الأصلي عبر تحويل عكسي. يتطلب بناء مثيل من هذه الفئة وسيطين: regressor الذي سيستخدم للتوقع، و transformer الذي سيتم تطبيقه على متغير الهدف.

فيما يلي مثال على استخدام TransformedTargetRegressor:

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)
X, y = X[:2000, :], y[:2000]  # select a subset of data
transformer = QuantileTransformer(output_distribution='normal')
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor, transformer=transformer)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regr.fit(X_train, y_train)
print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))
```

FeatureUnion:

تسمح فئة FeatureUnion بالدمج بين عدة أدوات تحويل إلى أداة تحويل جديدة تجمع بين الإخراج الخاص بها. تأخذ FeatureUnion قائمة من أدوات التحويل. خلال التركيب، يتم تركيب كل منها على البيانات بشكل مستقل. يتم تطبيق أدوات التحويل بالتوازي، ويتم جمع مصفوفات الميزات التي تنتجها إلى جانب بعضها البعض في مصفوفة أكبر.

فيما يلي مثال على استخدام FeatureUnion:

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
combined = FeatureUnion(estimators)
```

يمكن أيضًا استخدام ColumnTransformer للتعامل مع البيانات غير المتجانسة.
    ========================================

    تحتوي العديد من مجموعات البيانات على ميزات من أنواع مختلفة، مثل النصوص والأرقام العشرية والتواريخ، حيث يتطلب كل نوع من الميزات خطوات منفصلة لمعالجة البيانات أو استخراج الميزات. غالبًا ما يكون من الأسهل معالجة البيانات قبل تطبيق أساليب scikit-learn، على سبيل المثال باستخدام pandas. قد تكون معالجة بياناتك قبل تمريرها إلى scikit-learn مشكلة لأحد الأسباب التالية:

    1. دمج الإحصائيات من بيانات الاختبار في معالجات البيانات الأولية يجعل درجات التحقق المتقاطع غير موثوقة (تُعرف باسم تسرب البيانات)، على سبيل المثال في حالة مقياس النطاق أو إكمال القيم المفقودة.
    2. قد ترغب في تضمين معلمات معالجات البيانات الأولية في بحث المعلمات.

    يساعد ColumnTransformer في تنفيذ تحويلات مختلفة لأعمدة مختلفة من البيانات، ضمن Pipeline الذي يكون آمنًا من تسرب البيانات والذي يمكن معلمته. يعمل ColumnTransformer على المصفوفات والمصفوفات المتفرقة وإطارات بيانات pandas.

    يمكن تطبيق تحويل مختلف على كل عمود، مثل معالجة البيانات مسبقًا أو طريقة استخراج ميزة محددة::

        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'city': ['London', 'London', 'Paris', 'Sallisaw'],
        ...     'title': ["His Last Bow", "How Watson Learned the Trick",
        ...               "A Moveable Feast", "The Grapes of Wrath"],
        ...     'expert_rating': [5, 3, 4, 5],
        ...     'user_rating': [4, 5, 4, 3]})

    بالنسبة لهذه البيانات، قد نرغب في ترميز عمود "city" كمتغير فئوي باستخدام OneHotEncoder ولكن تطبيق CountVectorizer على عمود "title". نظرًا لأننا قد نستخدم طرق استخراج ميزات متعددة على نفس العمود، فإننا نعطي كل محول اسمًا فريدًا، لنقل "city_category" و "title_bow". بشكل افتراضي، يتم تجاهل أعمدة التصنيف المتبقية (remainder='drop')::

        >>> from sklearn.compose import ColumnTransformer
        >>> from sklearn.feature_extraction.text import CountVectorizer
        >>> from sklearn.preprocessing import OneHotEncoder
        >>> column_trans = ColumnTransformer(
        ...     [('categories', OneHotEncoder(dtype='int'), ['city']),
        ...      ('title_bow', CountVectorizer(), 'title')],
        ...     remainder='drop', verbose_feature_names_out=False)

        >>> column_trans.fit(X)
        ColumnTransformer(transformers=[('categories', OneHotEncoder(dtype='int'),
                                         ['city']),
                                        ('title_bow', CountVectorizer(), 'title')],
                          verbose_feature_names_out=False)

        >>> column_trans.get_feature_names_out()
        array(['city_London', 'city_Paris', 'city_Sallisaw', 'bow', 'feast',
               'grapes', 'his', 'how', 'last', 'learned', 'moveable', 'of', 'the',
                'trick', 'watson', 'wrath'], ...)

        >>> column_trans.transform(X).toarray()
        array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
               [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]]...)

    في المثال أعلاه، يتوقع CountVectorizer إدخالًا أحادي البعد وبالتالي تم تحديد الأعمدة كسلسلة (``'title'``). ومع ذلك، يتوقع OneHotEncoder مثل معظم المحولات الأخرى بيانات ثنائية الأبعاد، وبالتالي في هذه الحالة تحتاج إلى تحديد العمود كمصفوفة من السلاسل (``['city']``).

    بصرف النظر عن قيمة скаляр أو قائمة من عنصر واحد، يمكن تحديد اختيار العمود كقائمة من عناصر متعددة أو صفيف صحيح أو شريحة أو قناع منطقي أو باستخدام make_column_selector. يتم استخدام make_column_selector لتحديد الأعمدة بناءً على نوع البيانات أو اسم العمود::

        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.compose import make_column_selector
        >>> ct = ColumnTransformer([
        ...       ('scale', StandardScaler(),
        ...       make_column_selector(dtype_include=np.number)),
        ...       ('onehot',
        ...       OneHotEncoder(),
        ...       make_column_selector(pattern='city', dtype_include=object))])
        >>> ct.fit_transform(X)
        array([[ 0.904...,  0.      ,  1. ,  0. ,  0. ],
               [-1.507...,  1.414...,  1. ,  0. ,  0. ],
               [-0.301...,  0.      ,  0. ,  1. ,  0. ],
               [ 0.904..., -1.414...,  0. ,  0. ,  1. ]])

    يمكن أن تشير السلاسل إلى الأعمدة إذا كان الإدخال عبارة عن DataFrame، ويتم تفسير الأعداد الصحيحة دائمًا على أنها الأعمدة الموضعية.

    يمكننا الاحتفاظ بأعمدة التصنيف المتبقية عن طريق تعيين remainder='passthrough'. يتم إلحاق القيم بنهاية التحويل::

        >>> column_trans = ColumnTransformer(
        ...     [('city_category', OneHotEncoder(dtype='int'),['city']),
        ...      ('title_bow', CountVectorizer(), 'title')],
        ...     remainder='passthrough')

        >>> column_trans.fit_transform(X)
        array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 4],
               [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 3, 5],
               [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 4],
               [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 5, 3]]...)

    يمكن تعيين معامل remainder إلى مقدر لتحويل أعمدة التصنيف المتبقية. يتم إلحاق القيم المحولة بنهاية التحويل::

        >>> from sklearn.preprocessing import MinMaxScaler
        >>> column_trans = ColumnTransformer(
        ...     [('city_category', OneHotEncoder(), ['city']),
        ...      ('title_bow', CountVectorizer(), 'title')],
        ...     remainder=MinMaxScaler())

        >>> column_trans.fit_transform(X)[:, -2:]
        array([[1. , 0.5],
               [0. , 1. ],
               [0.5, 0.5],
               [1. , 0. ]])

    .. _make_column_transformer:

    تتوفر الدالة make_column_transformer لإنشاء كائن ColumnTransformer بسهولة أكبر. على وجه التحديد، سيتم إعطاء الأسماء تلقائيًا. ما يعادل المثال أعلاه سيكون::

```python
>>> from sklearn.compose import make_column_transformer
>>> column_trans = make_column_transformer(
...     (OneHotEncoder(), ['city']),
...     (CountVectorizer(), 'title'),
...     remainder=MinMaxScaler())
>>> column_trans
ColumnTransformer(remainder=MinMaxScaler(),
                 transformers=[('onehotencoder', OneHotEncoder(), ['city']),
                                ('countvectorizer', CountVectorizer(),
                                 'title')])
```

إذا تم تركيب :class:`~sklearn.compose.ColumnTransformer` باستخدام dataframe وكان للـ dataframe أسماء أعمدة نصية فقط، فإن تحويل dataframe سيستخدم أسماء الأعمدة لتحديد الأعمدة:

```python
>>> ct = ColumnTransformer(
...          [("scale", StandardScaler(), ["expert_rating"])]).fit(X)
>>> X_new = pd.DataFrame({"expert_rating": [5, 6, 1],
...                       "ignored_new_col": [1.2, 0.3, -0.1]})
>>> ct.transform(X_new)
array([[ 0.9...],
       [ 2.1...],
       [-3.9...]])
```

### تصور العناصر المركبة

(visualizing_composite_estimators: تصور العناصر المركبة)

  يتم عرض المقدرات (Estimators) بتمثيل HTML عند عرضها في دفتر ملاحظات الجوبتر (Jupyter notebook). هذا مفيد لتشخيص أو تصور خط أنابيب (Pipeline) مع العديد من المقدرات. يتم تنشيط هذا التصور افتراضيًا:

    ```python
    >>> column_trans  # doctest: +SKIP
    ```

    يمكن إلغاء تنشيطه عن طريق ضبط خيار "display" في الدالة `~sklearn.set_config` على 'text':

    ```python
    >>> from sklearn import set_config
    >>> set_config(display='text')  # doctest: +SKIP
    >>> # displays text representation in a jupyter context
    >>> column_trans  # doctest: +SKIP
    ```

    يمكن الاطلاع على مثال لإخراج HTML في قسم **HTML representation of Pipeline** من :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py`. كبديل، يمكن كتابة HTML إلى ملف باستخدام الدالة :func:`~sklearn.utils.estimator_html_repr`:

    ```python
    >>> from sklearn.utils import estimator_html_repr
    >>> with open('my_estimator.html', 'w') as f:  # doctest: +SKIP
    ...     f.write(estimator_html_repr(clf))
    ```

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_compose_plot_column_transformer.py`
    * :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py`
    

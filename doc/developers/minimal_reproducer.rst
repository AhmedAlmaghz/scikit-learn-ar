هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية:

.. _minimal_reproducer:

================================================
صياغة مثال قابل للتكرار والحد الأدنى لمكتبة scikit-learn
================================================


سواء كنت تقدم تقرير خطأ، أو تصمم مجموعة من الاختبارات، أو ببساطة تنشر سؤالًا في المناقشات، فإن القدرة على صياغة أمثلة قابلة للتكرار والحد الأدنى (أو أمثلة قابلة للتطبيق والحد الأدنى) هي المفتاح للتواصل بشكل فعال وكفء مع المجتمع.

توجد توجيهات جيدة جدًا على الإنترنت مثل `هذه الوثيقة من StackOverflow <https://stackoverflow.com/help/mcve>`_ أو `هذه التدوينة من Matthew Rocklin <https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_ حول صياغة أمثلة كاملة قابلة للتحقق (يشار إليها أدناه باسم MCVE). هدفنا ليس التكرار مع هذه المراجع، بل تقديم دليل خطوة بخطوة حول كيفية تقليص الخطأ حتى تصل إلى أقصر رمز ممكن لتكراره.

الخطوة الأولى قبل إرسال تقرير خطأ إلى مكتبة scikit-learn هي قراءة `قالب الإصدار <https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml>`_. إنه بالفعل غني بالمعلومات حول المعلومات التي سيُطلب منك تقديمها.


.. _good_practices:

ممارسات جيدة

    
(ملاحظة: لم تتم ترجمة الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية في النص الأصلي، وتم الحفاظ عليها كما هي في الترجمة.)

في هذا القسم، سنركز على قسم "خطوات/شيفرة لإعادة الإنتاج" من `قالب التقرير <https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml>`_. سنبدأ بمقطع شيفرة يوفر بالفعل مثالاً فاشلاً ولكن يمكن تحسين قابلية قراءته. سنصنع منه بعد ذلك مثالاً صغيراً وكاملاً ويمكن التحقق منه (MCVE).

**مثال**

.. code-block:: python

    # أنا أعمل حالياً في مشروع تعلم آلي وعندما حاولت ملائمة مثيل GradientBoostingRegressor لملف my_data.csv، حصلت على تحذير UserWarning:
    # "X يحتوي على أسماء الميزات، ولكن DecisionTreeRegressor تم ملائمته بدون أسماء الميزات". يمكنك الحصول على نسخة من مجموعتي البيانات من
    # https://example.com/my_data.csv والتحقق من أن ميزاتي تحتوي بالفعل على أسماء. يبدو أن المشكلة تظهر أثناء الملائمة عندما أستخدم قيمة عدد صحيح
    # للمعلمة n_iter_no_change.

    df = pd.read_csv('my_data.csv')
    X = df[["feature_name"]] # ميزاتي تحتوي بالفعل على أسماء
    y = df["target"]

    # نضبط random_state=42 لـ train_test_split
    X_train، X_test، y_train، y_test = train_test_split(
        X، y، test_size=0.33، random_state=42
    )

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # لا يثير المثيل ذو n_iter_no_change الافتراضي أي أخطاء أو تحذيرات
    gbdt = GradientBoostingRegressor(random_state=0)
    gbdt.fit(X_train، y_train)
    default_score = gbdt.score(X_test، y_test)

    # تظهر المشكلة عندما أغير القيمة لـ n_iter_no_change
    gbdt = GradientBoostingRegressor(random_state=0، n_iter_no_change=5)
    gbdt.fit(X_train، y_train)
    other_score = gbdt.score(X_test، y_test)

    other_score = gbdt.score(X_test، y_test)


قدم مثال شيفرة فاشل مع أقل قدر ممكن من التعليقات
----------------------------------------------------

غالباً ما يكون كتابة التعليمات لإعادة إنتاج المشكلة باللغة الإنجليزية غامضاً.
من الأفضل التأكد من توضيح جميع التفاصيل اللازمة لإعادة إنتاج المشكلة في مقتطف شيفرة بايثون لتجنب أي غموض. بالإضافة إلى ذلك، في هذه المرحلة، قدمت بالفعل وصفاً موجزاً في قسم **وصف المشكلة** من `قالب التقرير <https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml>`_.

الشيفرة التالية، على الرغم من أنها **لا تزال ليست الحد الأدنى**، **أفضل بكثير** لأنها يمكن نسخها ولصقها في طرفية بايثون لإعادة إنتاج المشكلة في خطوة واحدة. على وجه الخصوص:

- تحتوي على **جميع عبارات الاستيراد اللازمة**؛
- يمكنها جلب مجموعة البيانات العامة دون الحاجة إلى تنزيل ملف يدوياً ووضعه في الموقع المتوقع على القرص.

**مثال محسّن**

.. code-block:: python

    import pandas as pd

    df = pd.read_csv("https://example.com/my_data.csv")
    X = df[["feature_name"]]
    y = df["target"]

    from sklearn.model_selection import train_test_split

    X_train، X_test، y_train، y_test = train_test_split(
        X، y، test_size=0.33، random_state=42
    )

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.ensemble import GradientBoostingRegressor

    gbdt = GradientBoostingRegressor(random_state=0)
    gbdt.fit(X_train، y_train)  # لا يوجد تحذير
    default_score = gbdt.score(X_test، y_test)

    gbdt = GradientBoostingRegressor(random_state=0، n_iter_no_change=5)
    gbdt.fit(X_train، y_train)  # يثير تحذيراً
    other_score = gbdt.score(X_test، y_test)
    other_score = gbdt.score(X_test، y_test)


قلل برنامجك النصي إلى شيء صغير قدر الإمكان
-------------------------------------------------------

يجب أن تسأل نفسك أي سطور الشيفرة ذات صلة وأيها غير ذات صلة بإعادة إنتاج المشكلة. سيساعد حذف سطور الشيفرة غير الضرورية أو تبسيط استدعاءات الدالة عن طريق حذف الخيارات غير الافتراضية غير ذات الصلة في تضييق سبب المشكلة لك وللمساهمين الآخرين.

على وجه الخصوص، لهذا المثال المحدد:

- ليس للتحذير أي علاقة بـ `train_test_split` لأنه يظهر بالفعل في خطوة التدريب، قبل استخدام مجموعة الاختبار.
- وبالمثل، فإن السطور التي تحسب الدرجات على مجموعة الاختبار ليست ضرورية؛
- يمكن إعادة إنتاج المشكلة لأي قيمة لـ `random_state`، لذا اتركها كافتراضي؛
- يمكن إعادة إنتاج المشكلة دون معالجة مسبقة للبيانات باستخدام `StandardScaler`.

**مثال محسّن**

.. code-block:: python

    import pandas as pd
    df = pd.read_csv("https://example.com/my_data.csv")
    X = df[["feature_name"]]
    y = df["target"]

    from sklearn.ensemble import GradientBoostingRegressor

    gbdt = GradientBoostingRegressor()
    gbdt.fit(X، y)  # لا يوجد تحذير

    gbdt = GradientBoostingRegressor(n_iter_no_change=5)
    gbdt.fit(X، y)  # يثير تحذيراً


**لا** تبلغ عن بياناتك ما لم تكن ضرورية للغاية

الفكرة هي جعل الشفرة البرمجية مكتفية ذاتياً قدر الإمكان. للقيام بذلك، يمكنك استخدام :ref:`synth_data`. يمكن إنشاؤها باستخدام numpy أو pandas أو وحدة :mod:`sklearn.datasets`. في معظم الأحيان، لا يتعلق الخطأ بنية معينة لبياناتك. حتى إذا كان الأمر كذلك، فحاول العثور على مجموعة بيانات متاحة لها خصائص مشابهة لبياناتك وتعيد إنتاج المشكلة. في هذه الحالة بالتحديد، نحن مهتمون بالبيانات التي لها أسماء مميزة للميزات.

**مثال محسن**

.. code-block:: python

    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor

    df = pd.DataFrame(
        {
            "feature_name": [-12.32, 1.43, 30.01, 22.17],
            "target": [72, 55, 32, 43],
        }
    )
    X = df[["feature_name"]]
    y = df["target"]

    gbdt = GradientBoostingRegressor()
    gbdt.fit(X, y) # no warning
    gbdt = GradientBoostingRegressor(n_iter_no_change=5)
    gbdt.fit(X, y) # raises warning

كما ذكرنا سابقاً، فإن مفتاح التواصل هو قابلية قراءة الشفرة البرمجية، ويمكن أن يكون التنسيق الجيد ميزة إضافية حقاً. لاحظ أنه في المقتطف السابق:

- نحاول تقييد جميع الأسطر بحد أقصى 79 حرفاً لتجنب أشرطة التمرير الأفقية في كتل مقتطفات الشفرة البرمجية المقدمة في المشكلة على GitHub؛
- نستخدم أسطر فارغة لفصل مجموعات الوظائف ذات الصلة؛
- نضع جميع عمليات الاستيراد في مجموعتها الخاصة في البداية.

يمكن تنفيذ خطوات التبسيط المقدمة في هذا الدليل بترتيب مختلف عن التدرج الذي أظهرناه هنا. النقاط المهمة هي:

- يجب أن يكون أداة إعادة الإنتاج الدنيا قابلة للتشغيل عن طريق نسخ ولصق بسيط في محطة بايثون؛
- يجب تبسيطها قدر الإمكان عن طريق إزالة أي خطوات رمزية غير ضرورية تماماً لإعادة إنتاج المشكلة الأصلية؛
- من الناحية المثالية، يجب أن تعتمد فقط على مجموعة بيانات دنيا تم إنشاؤها على الطاير عن طريق تشغيل الشفرة البرمجية بدلاً من الاعتماد على بيانات خارجية، إن أمكن.


استخدم تنسيق Markdown
-----------------------

لتهيئة الشفرة البرمجية أو النص في كتلة مميزة خاصة به، استخدم علامات التنصيص الخلفية الثلاثية. يدعم `Markdown <https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax>`_ معرّف لغة اختياري لتمكين تمييز بناء الجملة في كتلة الشفرة البرمجية المسوّرة. على سبيل المثال::

    ```python
    from sklearn.datasets import make_blobs

    n_samples = 100
    n_components = 3
    X, y = make_blobs(n_samples=n_samples, centers=n_components)
    ```

سيؤدي هذا إلى عرض مقتطف منسق من بايثون على النحو التالي

.. code-block:: python

    from sklearn.datasets import make_blobs

    n_samples = 100
    n_components = 3
    X, y = make_blobs(n_samples=n_samples, centers=n_components)

ليس من الضروري إنشاء عدة كتل من الشفرة البرمجية عند الإبلاغ عن خلل. تذكر أن المراجعين الآخرين سيقومون بنسخ ولصق شفرة برمجية، وسيجعل وجود خلية واحدة مهمتهم أسهل.

في القسم المسمى **النتائج الفعلية** من `قالب المشكلة <https://github.com/scikit-learn/scikit-learn/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml>`_، يُطلب منك تقديم رسالة الخطأ بما في ذلك التتبع الكامل للاستثناء. في هذه الحالة، استخدم مؤهل `python-traceback`. على سبيل المثال::

    ```python-traceback
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-1-a674e682c281> in <module>
        4 vectorizer = CountVectorizer(input=docs, analyzer='word')
        5 lda_features = vectorizer.fit_transform(docs)
    ----> 6 lda_model = LatentDirichletAllocation(
        7     n_topics=10,
        8     learning_method='online',

    TypeError: __init__() got an unexpected keyword argument 'n_topics'
    ```

ينتج ما يلي عند عرضه:

.. code-block:: python

    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-1-a674e682c281> in <module>
        4 vectorizer = CountVectorizer(input=docs, analyzer='word')
        5 lda_features = vectorizer.fit_transform(docs)
    ----> 6 lda_model = LatentDirichletAllocation(
        7     n_topics=10,
        8     learning_method='online',

    TypeError: __init__() got an unexpected keyword argument 'n_topics'


.. _synth_data:

مجموعة البيانات الاصطناعية

    

قبل اختيار مجموعة بيانات اصطناعية معينة، يجب أولاً تحديد نوع المشكلة التي تحلها: هل هي تصنيف أم انحدار أم تجميع، إلخ؟

بمجرد تضييق نوع المشكلة، تحتاج إلى توفير مجموعة بيانات اصطناعية وفقًا لذلك. في معظم الأحيان، تحتاج فقط إلى مجموعة بيانات بسيطة. فيما يلي قائمة غير شاملة بالأدوات التي قد تساعدك.

NumPy
-----

يمكن استخدام أدوات NumPy مثل `numpy.random.randn <https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html>`_ و `numpy.random.randint <https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html>`_ لإنشاء بيانات رقمية وهمية.

- انحدار

  تأخذ الانحدارات البيانات الرقمية المستمرة كميزات وهدف.

  .. code-block:: python

      import numpy as np

      rng = np.random.RandomState(0)
      n_samples, n_features = 5, 5
      X = rng.randn(n_samples, n_features)
      y = rng.randn(n_samples)

يمكن استخدام جزء مماثل من التعليمات البرمجية كبيانات اصطناعية عند اختبار أدوات التحجيم مثل :class:`sklearn.preprocessing.StandardScaler`.

- تصنيف

  إذا لم يتم رفع الخطأ أثناء تشفير متغير فئوي، فيمكنك إطعام بيانات رقمية إلى مصنف. فقط تذكر التأكد من أن الهدف هو في الواقع عدد صحيح.

  .. code-block:: python

      import numpy as np

      rng = np.random.RandomState(0)
      n_samples, n_features = 5, 5
      X = rng.randn(n_samples, n_features)
      y = rng.randint(0, 2, n_samples)  # هدف ثنائي مع قيم في {0، 1}


  إذا حدث الخطأ فقط مع تسميات الفئة غير العددية، فقد ترغب في إنشاء هدف عشوائي باستخدام `numpy.random.choice <https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html>`_.

  .. code-block:: python

      import numpy as np

      rng = np.random.RandomState(0)
      n_samples, n_features = 50, 5
      X = rng.randn(n_samples, n_features)
      y = np.random.choice(
          ["male", "female", "other"], size=n_samples, p=[0.49, 0.49, 0.02]
      )

Pandas
------

تتوقع بعض كائنات scikit-learn dataframes pandas كمدخلات. في هذه الحالة، يمكنك تحويل صفائف numpy إلى كائنات pandas باستخدام `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_، أو `pandas.Series <https://pandas.pydata.org/docs/reference/api/pandas.Series.html>`_.

.. code-block:: python

    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(0)
    n_samples, n_features = 5, 5
    X = pd.DataFrame(
        {
            "continuous_feature": rng.randn(n_samples),
            "positive_feature": rng.uniform(low=0.0, high=100.0, size=n_samples),
            "categorical_feature": rng.choice(["a", "b", "c"], size=n_samples),
        }
    )
    y = pd.Series(rng.randn(n_samples))

بالإضافة إلى ذلك، يشتمل scikit-learn على العديد من :ref:`مولدات العينات <sample_generators>` التي يمكن استخدامها لبناء مجموعات بيانات اصطناعية ذات حجم ومعايير تحكم معقدة.

`make_regression`
-----------------

كما يوحي الاسم، تنتج :class:`sklearn.datasets.make_regression` أهداف انحدار مع ضوضاء كدالة خطية عشوائية من الميزات العشوائية.

.. code-block:: python

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=20)

`make_classification`
---------------------

تقوم :class:`sklearn.datasets.make_classification` بإنشاء مجموعات بيانات متعددة الفئات مع مجموعات Gaussian متعددة لكل فئة. يمكن إدخال الضوضاء عن طريق الميزات المرتبطة أو الزائدة أو غير الإعلامية.

.. code-block:: python

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1
    )

`make_blobs`
------------

على غرار `make_classification`، تنشئ :class:`sklearn.datasets.make_blobs` مجموعات بيانات متعددة الفئات باستخدام مجموعات من النقاط موزعة بشكل طبيعي. إنه يوفر تحكمًا أكبر فيما يتعلق بمراكز وانحرافات كل مجموعة، وبالتالي فهو مفيد لإثبات التجميع.

.. code-block:: python

    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=10, centers=3, n_features=2)

أدوات تحميل مجموعة البيانات
-------------------------

يمكنك استخدام :ref:`datasets <datasets>` لتحميل واسترداد العديد من مجموعات البيانات المرجعية الشائعة. هذا الخيار مفيد عندما يتعلق الخطأ بالهيكل الخاص للبيانات، على سبيل المثال التعامل مع القيم المفقودة أو التعرف على الصور.

.. code-block:: python

    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)

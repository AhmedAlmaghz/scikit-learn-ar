هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية كما هي:

.. _loading_other_datasets:

تحميل مجموعات بيانات أخرى

(ملاحظة: لم يتم ترجمة عنوان القسم لأنه يتضمن رمز خاص "_")

  هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية:

    ======================

    .. currentmodule:: sklearn.datasets

    .. _sample_images:

    صور العينات
    -------------

    تضم Scikit-learn أيضًا عددًا من صور JPEG النموذجية المنشورة تحت رخصة Creative Commons من قبل مؤلفيها. يمكن استخدام هذه الصور لاختبار الخوارزميات والأنابيب على بيانات ثنائية الأبعاد.

    .. autosummary::

       load_sample_images
       load_sample_image

    .. image:: ../auto_examples/cluster/images/sphx_glr_plot_color_quantization_001.png
       :target: ../auto_examples/cluster/plot_color_quantization.html
       :scale: 30
       :align: right


    .. warning::

      يستند الترميز الافتراضي للصور إلى النوع ``uint8`` لتوفير الذاكرة. غالبًا ما تعمل خوارزميات التعلم الآلي بشكل أفضل إذا تم تحويل الدخل إلى تمثيل بالنقطة العائمة أولاً. أيضًا، إذا كنت تخطط لاستخدام ``matplotlib.pyplpt.imshow``، فلا تنسَ القياس إلى النطاق 0 - 1 كما هو موضح في المثال التالي.

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_cluster_plot_color_quantization.py`

    .. _libsvm_loader:

    مجموعات البيانات بتنسيق svmlight / libsvm
    ------------------------------------

    تشمل scikit-learn وظائف مساعدة لتحميل مجموعات البيانات بتنسيق svmlight / libsvm. في هذا التنسيق، يأخذ كل سطر الشكل ``<label> <feature-id>:<feature-value> <feature-id>:<feature-value> ...``. هذا التنسيق مناسب بشكل خاص لمجموعات البيانات المتفرقة. في هذه الوحدة، يتم استخدام المصفوفات CSR المتفرقة لـ Scipy لـ ``X`` وتستخدم مصفوفات Numpy لـ ``y``.

    يمكنك تحميل مجموعة بيانات كما يلي::

      >>> from sklearn.datasets import load_svmlight_file
      >>> X_train, y_train = load_svmlight_file("/path/to/train_dataset.txt")
      ...                                                         # doctest: +SKIP

    يمكنك أيضًا تحميل مجموعتي بيانات (أو أكثر) في وقت واحد::

      >>> X_train, y_train, X_test, y_test = load_svmlight_files(
      ...     ("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
      ...                                                         # doctest: +SKIP

    في هذه الحالة، يضمن أن ``X_train`` و ``X_test`` لهما نفس عدد الميزات. طريقة أخرى لتحقيق نفس النتيجة هي إصلاح عدد الميزات::

      >>> X_test, y_test = load_svmlight_file(
      ...     "/path/to/test_dataset.txt", n_features=X_train.shape[1])
      ...                                                         # doctest: +SKIP

    .. rubric:: روابط ذات صلة

    - `مجموعات البيانات العامة بتنسيق svmlight / libsvm`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets
    - `تنفيذ أسرع متوافق مع واجهة برمجة التطبيقات`: https://github.com/mblondel/svmlight-loader

    ..
        For doctests:

        >>> import numpy as np
        >>> import os

    .. _openml:

    تنزيل مجموعات البيانات من مستودع openml.org

    (لاحظ أنني حافظت على التاجات والرموز الخاصة في الترجمة، ولكنني لم أترجم الروابط والتاجات والرموز الخاصة كما طلبت في التعليمات.)

`openml.org <https://openml.org>`_ هو مستودع عام لبيانات وتجارب التعلم الآلي، والذي يسمح للجميع بتحميل مجموعات البيانات المفتوحة.

يمكن لحزمة ``sklearn.datasets`` تنزيل مجموعات البيانات من المستودع باستخدام الدالة :func:`sklearn.datasets.fetch_openml`.

على سبيل المثال، لتنزيل مجموعة بيانات تعبيرات الجينات في أدمغة الفئران::

  >>> from sklearn.datasets import fetch_openml
  >>> mice = fetch_openml(name='miceprotein', version=4)

لتحديد مجموعة بيانات بالكامل، تحتاج إلى توفير اسم وإصدار، على الرغم من أن الإصدار اختياري، انظر :ref:`openml_versions` أدناه.
تحتوي مجموعة البيانات على ما مجموعه 1080 مثالًا ينتمون إلى 8 فئات مختلفة::

  >>> mice.data.shape
  (1080, 77)
  >>> mice.target.shape
  (1080,)
  >>> np.unique(mice.target)
 array(['c-CS-m', 'c-CS-s', 'c-SC-m', 'c-SC-s', 't-CS-m', 't-CS-s', 't-SC-m', 't-SC-s'], dtype=object)

يمكنك الحصول على مزيد من المعلومات حول مجموعة البيانات من خلال النظر في السمات ``DESCR`` و ``details``::

  >>> print(mice.DESCR) # doctest: +SKIP
  **المؤلف**: Clara Higuera, Katheleen J. Gardiner, Krzysztof J. Cios
  **المصدر**: [UCI](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) - 2015
  **الرجاء الاستشهاد**: Higuera C, Gardiner KJ, Cios KJ (2015) Self-Organizing
  Feature Maps Identify Proteins Critical to Learning in a Mouse Model of Down
  Syndrome. PLoS ONE 10(6): e0129126...

  >>> mice.details # doctest: +SKIP
  {'id': '40966', 'name': 'MiceProtein', 'version': '4', 'format': 'ARFF',
  'upload_date': '2017-11-08T16:00:15', 'licence': 'Public',
  'url': 'https://www.openml.org/data/v1/download/17928620/MiceProtein.arff',
  'file_id': '17928620', 'default_target_attribute': 'class',
  'row_id_attribute': 'MouseID',
  'ignore_attribute': ['Genotype', 'Treatment', 'Behavior'],
  'tag': ['OpenML-CC18', 'study_135', 'study_98', 'study_99'],
  'visibility': 'public', 'status': 'active',
  'md5_checksum': '3c479a6885bfa0438971388283a1ce32'}


تحتوي السمة ``DESCR`` على وصف نصي مجاني للبيانات، بينما تحتوي ``details`` على قاموس من البيانات الوصفية المخزنة بواسطة openml، مثل معرف مجموعة البيانات.
لمزيد من التفاصيل، راجع `وثائق OpenML <https://docs.openml.org/#data>`_ معرف ``data_id`` لمجموعة بيانات بروتين الفئران هو 40966، ويمكنك استخدام هذا (أو الاسم) للحصول على مزيد من المعلومات حول مجموعة البيانات على موقع openml::

  >>> mice.url
  'https://www.openml.org/d/40966'

يحدد ``data_id`` أيضًا مجموعة بيانات بشكل فريد من OpenML::

  >>> mice = fetch_openml(data_id=40966)
  >>> mice.details # doctest: +SKIP
  {'id': '4550', 'name': 'MiceProtein', 'version': '1', 'format': 'ARFF',
  'creator': ...,
  'upload_date': '2016-02-17T14:32:49', 'licence': 'Public', 'url':
  'https://www.openml.org/data/v1/download/1804243/MiceProtein.ARFF', 'file_id':
  '1804243', 'default_target_attribute': 'class', 'citation': 'Higuera C,
  Gardiner KJ, Cios KJ (2015) Self-Organizing Feature Maps Identify Proteins
  Critical to Learning in a Mouse Model of Down Syndrome. PLoS ONE 10(6):
  e0129126. [Web Link] journal.pone.0129126', 'tag': ['OpenML100', 'study_14',
  'study_34'], 'visibility': 'public', 'status': 'active', 'md5_checksum':
  '3c479a6885bfa0438971388283a1ce32'}

.. _openml_versions:

إصدارات مجموعة البيانات
~~~~~~~~~~~~~~~~

يتم تحديد مجموعة البيانات بشكل فريد بواسطة ``data_id`` الخاص بها، ولكن ليس بالضرورة بالاسم. يمكن أن توجد العديد من "الإصدارات" المختلفة لمجموعة بيانات بنفس الاسم والتي يمكن أن تحتوي على مجموعات بيانات مختلفة تمامًا.
إذا تم العثور على إصدار معين من مجموعة بيانات تحتوي على مشكلات كبيرة، فقد يتم إلغاء تنشيطه. قد يؤدي استخدام اسم لتحديد مجموعة بيانات إلى نتائج مختلفة في أوقات مختلفة إذا أصبحت الإصدارات السابقة غير نشطة.
يمكنك أن ترى أن مجموعة البيانات التي تحتوي على ``data_id`` 40966 التي قمنا بجلبها أعلاه هي الإصدار الأول من مجموعة بيانات "miceprotein"::

  >>> mice.details['version']  #doctest: +SKIP
  '1'

في الواقع، يحتوي هذا dataset على إصدار واحد فقط. مجموعة بيانات iris من ناحية أخرى لديها إصدارات متعددة::

  >>> iris = fetch_openml(name="iris")
  >>> iris.details['version']  #doctest: +SKIP
  '1'
  >>> iris.details['id']  #doctest: +SKIP
  '61'

  >>> iris_61 = fetch_openml(data_id=61)
  >>> iris_61.details['version']
  '1'
  >>> iris_61.details['id']
  '61'

  >>> iris_969 = fetch_openml(data_id=969)
  >>> iris_969.details['version']
  '3'
  >>> iris_969.details['id']
  '969'

يؤدي تحديد مجموعة البيانات بالاسم "iris" إلى الحصول على أدنى إصدار، الإصدار 1، مع ``data_id`` 61. للتأكد من حصولك دائمًا على هذه المجموعة المحددة من البيانات، من الأفضل تحديدها بواسطة ``data_id`` لمجموعة البيانات. المجموعة الأخرى من البيانات، مع ``data_id`` 969، هي الإصدار 3 (أصبح الإصدار 2 غير نشط)، وتحتوي على إصدار ثنائي من البيانات::

  >>> np.unique(iris_969.target)
 array(['N', 'P'], dtype=object)

يمكنك أيضًا تحديد كلاً من الاسم والإصدار، والذي يحدد أيضًا مجموعة البيانات بشكل فريد::

  >>> iris_version_3 = fetch_openml(name="iris", version=3)
  >>> iris_version_3.details['version']
  '3'
  >>> iris_version_3.details['id']
  '969'


.. rubric:: المراجع

* :arxiv:`Vanschoren, van Rijn, Bischl and Torgo. "OpenML: networked science in
 machine learning" ACM SIGKDD Explorations Newsletter, 15(2), 49-60, 2014.
  <1407.7722>`

.. _openml_parser:

محلل ARFF
~~~~~~~~~

من الإصدار 1.2، توفر scikit-learn وسيطة كلمة رئيسية جديدة `parser` توفر العديد من الخيارات لتحليل ملفات ARFF التي يوفرها OpenML. يستند المحلل القديم (أي `parser="liac-arff"`) إلى المشروع `LIAC-ARFF <https://github.com/renatopp/liac-arff>`_. ومع ذلك، فإن هذا المحلل بطيء ويستهلك مزيدًا من
يؤدي استخدام محللي "pandas" و "liac-arff" إلى اختلافات في أنواع البيانات في المخرجات. فيما يلي بعض الاختلافات البارزة:

- يقوم محلل "liac-arff" دائمًا بترميز الميزات النوعية ككائنات من نوع "str". على العكس من ذلك، يستنتج محلل "pandas" النوع أثناء القراءة وسيتم تحويل الفئات الرقمية إلى أعداد صحيحة كلما كان ذلك ممكنًا.
- يستخدم محلل "liac-arff" النوع float64 لترميز الميزات الرقمية الموسومة بـ 'REAL' و 'NUMERICAL' في البيانات الوصفية. بينما يستنتج محلل "pandas" ما إذا كانت هذه الميزات الرقمية تتوافق مع أعداد صحيحة ويستخدم نوع البيانات Integer التابع لـ pandas.
- على وجه الخصوص، يتم عادةً تحميل مجموعات بيانات التصنيف ذات الفئات الصحيحة بهذه الطريقة (0، 1، ...) مع محلل "pandas" بينما سيجبر محلل "liac-arff" على استخدام تسميات الفئات المشفرة كـ "str" مثل "0"، "1" وما إلى ذلك.
- لن يقم محلل "pandas" بإزالة علامات الاقتباس الفردية - أي ' - من أعمدة النص. على سبيل المثال، سيتم الاحتفاظ بالنص 'my string' كما هو، بينما سيقوم محلل "liac-arff" بإزالة علامات الاقتباس الفردية. بالنسبة للأعمدة النوعية، تتم إزالة علامات الاقتباس الفردية من القيم.

بالإضافة إلى ذلك، عند استخدام as_frame=False، يعيد محلل "liac-arff" البيانات المشفرة بطريقة ترتيبية حيث يتم توفير الفئات في الخاصية "categories" لعناصر "bunch". بينما يعيد محلل "pandas" مصفوفة NumPy حيث الفئات. ثم يعود الأمر إلى المستخدم لتصميم خط أنابيب هندسة الميزات مع مثيل من OneHotEncoder أو OrdinalEncoder، وغالبًا ما يكون ملفوفًا في ColumnTransformer، لمعالجة الأعمدة النوعية بشكل صريح. راجع على سبيل المثال: :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py`.

تحميل من مجموعات بيانات خارجية

(لا حاجة لترجمة الروابط والأكواد البرمجية)

  هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة، والرموز والمعادلات الرياضية، والروابط، والتاجات، والشفرة البرمجية كما هي:

------------------------------

يعمل scikit-learn على أي بيانات رقمية مخزنة كصفائف numpy أو مصفوفات scipy sparse. كما أنه يقبل أنواعًا أخرى قابلة للتحويل إلى صفائف رقمية مثل pandas DataFrame.

فيما يلي بعض الطرق الموصى بها لتحميل البيانات العمودية القياسية إلى تنسيق قابل للاستخدام بواسطة scikit-learn:

* يوفر pandas.io <https://pandas.pydata.org/pandas-docs/stable/io.html> أدوات لقراءة البيانات من التنسيقات الشائعة بما في ذلك CSV و Excel و JSON و SQL. يمكن أيضًا إنشاء DataFrames من قوائم التوائم أو القواميس. يتعامل Pandas بسلاسة مع البيانات غير المتجانسة ويوفر أدوات للتلاعب والتحويل إلى صفيف رقمي مناسب لـ scikit-learn.
* يتخصص scipy.io <https://docs.scipy.org/doc/scipy/reference/io.html> في التنسيقات الثنائية المستخدمة غالبًا في سياق الحوسبة العلمية مثل .mat و .arff
* numpy/routines.io <https://docs.scipy.org/doc/numpy/reference/routines.io.html> لتحميل البيانات العمودية القياسية في صفائف numpy
* функция load_svmlight_file من scikit-learn لتنسيق svmlight أو libSVM
* وظيفة load_files من scikit-learn لمجلدات ملفات النص حيث يكون اسم كل مجلد هو اسم كل فئة وكل ملف داخل كل مجلد يتوافق مع عينة واحدة من هذه الفئة

لبعض البيانات المتنوعة مثل الصور ومقاطع الفيديو والصوت، قد ترغب في الرجوع إلى:

* skimage.io <https://scikit-image.org/docs/dev/api/skimage.io.html> أو Imageio <https://imageio.readthedocs.io/en/stable/reference/core_v3.html> لتحميل الصور ومقاطع الفيديو في صفائف numpy
* scipy.io.wavfile.read <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html> لقراءة ملفات WAV في صفيف numpy

ستحتاج الميزات التصنيفية (أو الاسمية) المخزنة كسلاسل (شائعة في pandas DataFrames) إلى تحويلها إلى ميزات رقمية باستخدام OneHotEncoder أو OrdinalEncoder أو ما شابه. انظر preprocessing.

ملاحظة: إذا كنت تدير بياناتك الرقمية الخاصة، فمن المستحسن استخدام تنسيق ملف محسّن مثل HDF5 لتقليل أوقات تحميل البيانات. توفر مكتبات مختلفة مثل H5Py و PyTables و pandas واجهة Python لقراءة وكتابة البيانات في هذا التنسيق.

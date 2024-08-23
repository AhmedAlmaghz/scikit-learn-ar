.. currentmodule:: sklearn.model_selection

.. _grid_search:

===========================================
ضبط المعلمات الفوقية لمقدر
===========================================

المعلمات الفوقية هي معلمات لا يتم تعلمها مباشرة داخل المقدرات. في سكيت-ليرن يتم تمريرها كوسيطات إلى منشئ فئات المقدرات. تتضمن الأمثلة النموذجية ``C`` و ``kernel`` و ``gamma`` لمصنف المتجهات الداعمة، و ``alpha`` للasso، وما إلى ذلك.

من الممكن والموصى به البحث في مساحة المعلمات الفوقية للحصول على أفضل درجة :ref:`cross validation <cross_validation>`.

يمكن تحسين أي معلمة يتم توفيرها عند إنشاء مقدر بهذه الطريقة. على وجه التحديد، للعثور على الأسماء والقيم الحالية لجميع المعلمات لمقدر معين، استخدم::

  estimator.get_params()

يتكون البحث من:

- مقدر (مصنف أو مصنف مثل ``sklearn.svm.SVC()``)؛
- مساحة المعلمة؛
- طريقة للبحث أو أخذ عينات من المرشحين؛
- مخطط التحقق المتقاطع؛ و
- :ref:`score function <gridsearch_scoring>`.

يتم توفير نهجين عامين للبحث عن المعلمات في سكيت-ليرن: للقيم المحددة، :class:`GridSearchCV` يفكر بشكل شامل في جميع مجموعات المعلمات، بينما :class:`RandomizedSearchCV` يمكنه أخذ عينة من عدد معين من المرشحين من مساحة المعلمة بتوزيع محدد. لكل من هذه الأدوات نظراء تقليل النصف :class:`HalvingGridSearchCV` و :class:`HalvingRandomSearchCV`، والتي يمكن أن تكون أسرع بكثير في العثور على مجموعة جيدة من المعلمات.

بعد وصف هذه الأدوات، نوضح بالتفصيل :ref:`best practices <grid_search_tips>` المطبقة على هذه الأساليب. تسمح بعض النماذج باستراتيجيات بحث معلمات متخصصة وفعالة، تم تحديدها في :ref:`alternative_cv`.

لاحظ أنه من الشائع أن مجموعة فرعية صغيرة من هذه المعلمات يمكن أن يكون لها تأثير كبير على الأداء التنبئي أو الحسابي للنموذج بينما يمكن ترك الأخرى لقيمها الافتراضية. يوصى بقراءة سلسلة وثائق فئة المقدرة للحصول على فهم أدق لسلوكها المتوقع، ربما من خلال قراءة المرجع المرفق إلى الأدبيات.

البحث الشامل في الشبكة
======================

يتم توفير البحث في الشبكة بواسطة :class:`GridSearchCV` الذي يولد بشكل شامل المرشحين من شبكة قيم المعلمات المحددة باستخدام معلمة ``param_grid``. على سبيل المثال، ``param_grid`` التالي::

 param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
   ]

يحدد أنه يجب استكشاف شبكتين: واحدة مع نواة خطية وقيم C في [1، 10، 100، 1000]، والثانية مع نواة RBF، والمنتج المتقاطع لقيم C في [1، 10، 100، 1000] وقيم جاما في [0.001، 0.0001].

تنفذ مثيل :class:`GridSearchCV` واجهة برمجة التطبيقات المعتادة للمقدر: عند "تناسبها" على مجموعة بيانات، يتم تقييم جميع مجموعات قيم المعلمات الممكنة ويتم الاحتفاظ بأفضل مجموعة.

.. currentmodule:: sklearn.model_selection

.. rubric:: أمثلة

- انظر :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py` للحصول على مثال لحساب البحث في الشبكة على مجموعة بيانات الأرقام.

- انظر :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_text_feature_extraction.py` للحصول على مثال لربط بحث الشبكة معلمات من مستخرج ميزة مستندات نصية (عداد ن-غرام ومحول TF-IDF) مع مصنف (هنا SVM خطي مدرب مع SGD مع إما شبكة مرنة أو عقوبة L2) باستخدام مثيل :class:`~sklearn.pipeline.Pipeline`.

- انظر :ref:`sphx_glr_auto_examples_model_selection_plot_nested_cross_validation_iris.py` للحصول على مثال لبحث الشبكة داخل حلقة التحقق المتقاطع على مجموعة بيانات iris. هذه هي أفضل ممارسة لتقييم أداء النموذج مع بحث الشبكة.

- انظر :ref:`sphx_glr_auto_examples_model_selection_plot_multi_metric_evaluation.py` للحصول على مثال لـ :class:`GridSearchCV` المستخدمة لتقييم مقاييس متعددة في وقت واحد.

- انظر :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_refit_callable.py` للحصول على مثال لاستخدام واجهة ``refit=callable`` في :class:`GridSearchCV`. يوضح المثال كيف تضيف هذه الواجهة قدرًا معينًا من المرونة في تحديد "أفضل" مقدر. يمكن أيضًا استخدام هذه الواجهة في تقييم مقاييس متعددة.

- انظر :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_stats.py` للحصول على مثال لكيفية إجراء مقارنة إحصائية على مخرجات :class:`GridSearchCV`.

.. _randomized_parameter_search:

تحسين المعلمة العشوائية

في حين أن استخدام شبكة من إعدادات المعلمات هو حاليًا الطريقة الأكثر استخدامًا لتحسين المعلمات، إلا أن هناك طرق بحث أخرى لها خصائص أكثر ملاءمة.

تقوم RandomizedSearchCV بتنفيذ بحث عشوائي على المعلمات، حيث يتم أخذ كل إعداد من توزيع على قيم المعلمات المحتملة.

لهذا الأمر فائدتان رئيسيتان مقارنة بالبحث الشامل:

* يمكن اختيار ميزانية مستقلة عن عدد المعلمات والقيم المحتملة.
* إضافة معلمات لا تؤثر على الأداء لا تقلل من الكفاءة.

يتم تحديد كيفية أخذ عينات المعلمات باستخدام قاموس، يشبه جدًا تحديد المعلمات لـ GridSearchCV. بالإضافة إلى ذلك، يتم تحديد ميزانية الحوسبة، وهي عدد المرشحين الذين تم أخذ عينات منهم أو تكرارات أخذ العينات، باستخدام معلمة "n_iter".

بالنسبة لكل معلمة، يمكن تحديد توزيع على القيم المحتملة أو قائمة من الخيارات المنفصلة (والتي سيتم أخذ عينات منها بشكل موحد)::

  {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
    'kernel': ['rbf'], 'class_weight':['balanced', None]}

يستخدم هذا المثال وحدة "scipy.stats" التي تحتوي على العديد من التوزيعات المفيدة لأخذ عينات المعلمات، مثل "expon" و "gamma" و "uniform" و "loguniform" أو "randint".

من حيث المبدأ، يمكن تمرير أي وظيفة توفر طريقة "rvs" (عينة متغير عشوائي) لأخذ عينة من قيمة. يجب أن توفر المكالمة إلى وظيفة "rvs" عينات عشوائية مستقلة من قيم المعلمات المحتملة في المكالمات المتتالية.

.. تحذير::

    لا تسمح التوزيعات في "scipy.stats" قبل الإصدار scipy 0.16 بتحديد حالة عشوائية. بدلاً من ذلك، فإنها تستخدم حالة عشوائية عالمية في numpy، والتي يمكن زرعها عبر "np.random.seed" أو تعيينها باستخدام "np.random.set_state". ومع ذلك، بدءًا من scikit-learn 0.18، تقوم وحدة "sklearn.model_selection" بتعيين الحالة العشوائية التي يوفرها المستخدم إذا كان scipy >= 0.16 متاحًا أيضًا.

بالنسبة للمعلمات المستمرة، مثل "C" أعلاه، من المهم تحديد توزيع مستمر للاستفادة الكاملة من التوزيع العشوائي. بهذه الطريقة، ستؤدي زيادة "n_iter" دائمًا إلى بحث أدق.

المتغير العشوائي المستمر الموحد هو النسخة المستمرة من معلمة ذات مسافات لوغاريتمية. على سبيل المثال، لتحديد ما يعادل "C" من الأعلى، يمكن استخدام "loguniform(1, 100)" بدلاً من "[1, 10, 100]".

عكس المثال أعلاه في البحث الشبكي، يمكننا تحديد متغير عشوائي مستمر موزع بشكل موحد بين "1e0" و "1e3"::

 from sklearn.utils.fixes import loguniform
  {'C': loguniform(1e0, 1e3),
   'gamma': loguniform(1e-4, 1e-3),
   'kernel': ['rbf'],
   'class_weight':['balanced', None]}

.. أمثلة

* :ref:`sphx_glr_auto_examples_model_selection_plot_randomized_search.py` يقارن استخدام وكفاءة البحث العشوائي والبحث الشبكي.

.. المراجع

* Bergstra، J. و Bengio، Y.،
  البحث العشوائي لتحسين المعلمات الفوقية،
  The Journal of Machine Learning Research (2012)

.. _successive_halving_user_guide:

البحث عن المعلمات المثلى مع النصف المتتالي
    
(ملاحظة: لم يتم ترجمة القسم الأخير لأنه لا يحتوي على نص للترجمة، بل هو عنوان فرعي فقط.)

توفر Scikit-learn أيضًا مقدرات :class:`HalvingGridSearchCV` و :class:`HalvingRandomSearchCV` التي يمكن استخدامها للبحث في مساحة المعلمة باستخدام النصف المتتالي [1]_ [2]_. النصف المتتالي (SH) يشبه البطولة بين مجموعات المعلمات المرشحة. SH هي عملية اختيار تكرارية حيث يتم تقييم جميع المرشحين (مجموعات المعلمات) بموارد صغيرة في التكرار الأول. يتم اختيار عدد قليل فقط من هؤلاء المرشحين للتكرار التالي، والذي سيتم تخصيص المزيد من الموارد له. بالنسبة لضبط المعلمة، يكون المورد عادةً عدد عينات التدريب، ولكن يمكن أن يكون أيضًا معلمة رقمية عشوائية مثل `n_estimators` في الغابة العشوائية.

.. note::

    يجب أن يكون زيادة المورد المختار كبيرًا بما يكفي بحيث يتم الحصول على تحسن كبير في الدرجات عند أخذ الأهمية الإحصائية في الاعتبار.

كما هو موضح في الشكل أدناه، فقط مجموعة فرعية من المرشحين "تنجو" حتى التكرار الأخير. هؤلاء هم المرشحون الذين احتلوا باستمرار مرتبة بين المرشحين الأعلى تصنيفًا عبر جميع التكرارات. يتم تخصيص كمية متزايدة من الموارد لكل مرشح في كل تكرار، وهنا عدد العينات.

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_successive_halving_iterations_001.png
   :target: ../auto_examples/model_selection/plot_successive_halving_iterations.html
   :align: center

نصف هنا بإيجاز المعلمات الرئيسية، ولكن يتم وصف كل معلمة وتفاعلاتها بمزيد من التفاصيل في الأقسام أدناه. يتحكم معامل ``factor`` (> 1) في معدل نمو الموارد، ومعدل انخفاض عدد المرشحين. في كل تكرار، يتم ضرب عدد الموارد لكل مرشح في ``factor`` ويتم تقسيم عدد المرشحين على نفس العامل. إلى جانب ``resource`` و ``min_resources``، فإن ``factor`` هو أهم معلمة للتحكم في البحث في التنفيذ الخاص بنا، على الرغم من أن قيمة 3 عادةً ما تعمل بشكل جيد. يتحكم ``factor`` بشكل فعال في عدد التكرارات في :class:`HalvingGridSearchCV` وعدد المرشحين (افتراضيًا) والتكرارات في :class:`HalvingRandomSearchCV`. يمكن أيضًا استخدام ``aggressive_elimination=True`` إذا كان عدد الموارد المتاحة صغيرًا. مزيد من التحكم متاح من خلال ضبط معلمة ``min_resources``.

لا تزال هذه المقدرات **تجريبية**: قد تتغير تنبؤاتها وواجهة برمجة التطبيقات الخاصة بها دون أي دورة إهمال. لاستخدامها، تحتاج إلى استيراد ``enable_halving_search_cv`` بشكل صريح::

  >>> # explicitly require this experimental feature
  >>> from sklearn.experimental import enable_halving_search_cv  # noqa
  >>> # now you can import normally from model_selection
  >>> from sklearn.model_selection import HalvingGridSearchCV
  >>> from sklearn.model_selection import HalvingRandomSearchCV

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_model_selection_plot_successive_halving_heatmap.py`
* :ref:`sphx_glr_auto_examples_model_selection_plot_successive_halving_iterations.py`

اختيار ``min_resources`` وعدد المرشحين
    
[1]_ Jamieson, K., & Talwalkar, A. (2016). Non-stochastic best arm identification and hyperparameter optimization. In International Conference on Machine Learning (pp. 241-250). PMLR.

[2]_ Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). Hyperband: A novel bandit-based approach to hyperparameter optimization. In International Conference on Learning Representations.

بجانب "factor"، هناك اثنان من المعلمات الرئيسية التي تؤثر على سلوك البحث المتتالي بالنصف هما معلمة "min_resources" وعدد المرشحين (أو مجموعات المعلمات) التي يتم تقييمها. "min_resources" هي كمية الموارد المخصصة في التكرار الأول لكل مرشح. يتم تحديد عدد المرشحين مباشرة في :class:`HalvingRandomSearchCV`، ويتم تحديده من معلمة "param_grid" في :class:`HalvingGridSearchCV`.

فكر في حالة حيث المورد هو عدد العينات، ولدينا 1000 عينة. من الناحية النظرية، مع "min_resources=10" و "factor=2"، نحن قادرون على تشغيل **على الأكثر** 7 تكرارات مع الأرقام التالية من العينات: ``[10, 20, 40, 80, 160, 320, 640]``.

ولكن اعتمادًا على عدد المرشحين، قد نعمل أقل من 7 تكرارات: إذا بدأنا بعدد **صغير** من المرشحين، فقد يستخدم التكرار الأخير أقل من 640 عينة، مما يعني عدم استخدام جميع الموارد المتاحة (العينات). على سبيل المثال، إذا بدأنا بـ 5 مرشحين، فإننا نحتاج فقط إلى تكرارين: 5 مرشحين للتكرار الأول، ثم `5 // 2 = 2` مرشح في التكرار الثاني، وبعد ذلك نعرف أي مرشح يؤدي بشكل أفضل (لذلك لا نحتاج إلى تكرار ثالث). سنستخدم فقط 20 عينة على الأكثر وهو إهدار لأن لدينا 1000 عينة تحت تصرفنا. من ناحية أخرى، إذا بدأنا بعدد **كبير** من المرشحين، فقد ينتهي بنا الأمر مع العديد من المرشحين في التكرار الأخير، مما قد لا يكون مثاليًا دائمًا: فهذا يعني أن العديد من المرشحين سيعملون مع الموارد الكاملة، مما يقلل بشكل أساسي الإجراء إلى بحث قياسي.

في حالة :class:`HalvingRandomSearchCV`، يتم تعيين عدد المرشحين افتراضيًا بحيث يستخدم التكرار الأخير أكبر قدر ممكن من الموارد المتاحة. بالنسبة إلى :class:`HalvingGridSearchCV`، يتم تحديد عدد المرشحين بواسطة معلمة `param_grid`. سيؤثر تغيير قيمة "min_resources" على عدد التكرارات الممكنة، ونتيجة لذلك سيكون له أيضًا تأثير على العدد المثالي للمرشحين.

اعتبار آخر عند اختيار "min_resources" هو ما إذا كان من السهل التمييز بين المرشحين الجيدين والسيئين مع كمية صغيرة من الموارد. على سبيل المثال، إذا كنت بحاجة إلى الكثير من العينات للتمييز بين المعلمات الجيدة والسيئة، يوصى باستخدام "min_resources" عالية. من ناحية أخرى، إذا كان التمييز واضحًا حتى مع كمية صغيرة من العينات، فقد تكون "min_resources" الصغيرة مفضلة لأنها ستسرع الحساب.

لاحظ في المثال أعلاه أن التكرار الأخير لا يستخدم الحد الأقصى من الموارد المتاحة: 1000 عينة متاحة، ولكن يتم استخدام 640 فقط، على الأكثر. بشكل افتراضي، تحاول كل من :class:`HalvingRandomSearchCV` و :class:`HalvingGridSearchCV` استخدام أكبر عدد ممكن من الموارد في التكرار الأخير، مع القيد بأن كمية الموارد هذه يجب أن تكون مضاعفة لكل من `min_resources` و `factor` (سيكون هذا القيد واضحًا في القسم التالي). تحقق :class:`HalvingRandomSearchCV` ذلك عن طريق أخذ عينات من العدد الصحيح من المرشحين، بينما تحقق :class:`HalvingGridSearchCV` ذلك عن طريق تعيين `min_resources` بشكل صحيح. يرجى الاطلاع على :ref:`exhausting_the_resources` للتفاصيل.

.. _amount_of_resource_and_number_of_candidates:

كمية الموارد وعدد المرشحين في كل تكرار

في أي تكرار `i`، يتم تخصيص كمية معينة من الموارد لكل مرشح والتي نرمز لها بـ `n_resources_i`. يتم التحكم في هذه الكمية بواسطة المعلمتين ``factor`` و ``min_resources`` على النحو التالي (`factor` أكبر من 1 بشكل صارم)::

    n_resources_i = factor**i * min_resources,

أو ما يعادله::

    n_resources_{i+1} = n_resources_i * factor

حيث ``min_resources == n_resources_0`` هي كمية الموارد المستخدمة في التكرار الأول. يحدد ``factor`` أيضًا نسب المرشحين الذين سيتم اختيارهم للتكرار التالي::

    n_candidates_i = n_candidates // (factor ** i)

أو ما يعادله::

    n_candidates_0 = n_candidates
    n_candidates_{i+1} = n_candidates_i // factor

لذلك في التكرار الأول، نستخدم موارد ``min_resources`` ``n_candidates`` مرات. في التكرار الثاني، نستخدم موارد ``min_resources * factor`` ``n_candidates // factor`` مرات. في التكرار الثالث، نضاعف مرة أخرى الموارد لكل مرشح ونقسم عدد المرشحين. تتوقف هذه العملية عندما يتم الوصول إلى الحد الأقصى لكمية الموارد لكل مرشح، أو عندما نحدد أفضل مرشح. يتم تحديد أفضل مرشح في التكرار الذي يقيم `factor` أو أقل من المرشحين (انظر أدناه للحصول على شرح).

فيما يلي مثال باستخدام ``min_resources=3`` و ``factor=2``، بدءًا من 70 مرشحًا:

+-----------------------+-----------------------+
| ``n_resources_i``     | ``n_candidates_i``    |
+=======================+=======================+
| 3 (=min_resources)    | 70 (=n_candidates)    |
+-----------------------+-----------------------+
| 3 * 2 = 6             | 70 // 2 = 35          |
+-----------------------+-----------------------+
| 6 * 2 = 12            | 35 // 2 = 17          |
+-----------------------+-----------------------+
| 12 * 2 = 24           | 17 // 2 = 8           |
+-----------------------+-----------------------+
| 24 * 2 = 48           | 8 // 2 = 4            |
+-----------------------+-----------------------+
| 48 * 2 = 96           | 4 // 2 = 2            |
+-----------------------+-----------------------+

يمكننا ملاحظة أن:

- تتوقف العملية في التكرار الأول الذي يقيم `factor=2` من المرشحين: أفضل مرشح هو الأفضل من بين هذين المرشحين. ليس من الضروري تشغيل تكرار إضافي، نظرًا لأنه سيقيم مرشحًا واحدًا فقط (وهو أفضل مرشح، والذي حددناه بالفعل). لهذا السبب، بشكل عام، نريد أن يقوم التكرار الأخير بتقييم ``factor`` مرشحين على الأكثر. إذا قام التكرار الأخير بتقييم أكثر من `factor` مرشح، فإن هذا التكرار الأخير يتقلص إلى بحث منتظم (كما هو الحال في :class:`RandomizedSearchCV` أو :class:`GridSearchCV`).
- كل ``n_resources_i`` هو مضاعف لكل من ``factor`` و ``min_resources`` (والذي يتم تأكيده من خلال تعريفه أعلاه).

يمكن العثور على كمية الموارد المستخدمة في كل تكرار في السمة `n_resources_`.

اختيار مورد
-----------

بشكل افتراضي، يتم تعريف المورد من حيث عدد العينات. أي أن كل تكرار سيستخدم كمية متزايدة من العينات للتدريب عليها. ومع ذلك، يمكنك تحديد معلمة يدويًا لاستخدامها كمورد باستخدام معلمة ``resource``. فيما يلي مثال حيث يتم تعريف المورد من حيث عدد مقدرات الغابة العشوائية::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.experimental import enable_halving_search_cv  # noqa
    >>> from sklearn.model_selection import HalvingGridSearchCV
    >>> import pandas as pd
    >>>
    >>> param_grid = {'max_depth': [3, 5, 10],
    ...               'min_samples_split': [2, 5, 10]}
    >>> base_estimator = RandomForestClassifier(random_state=0)
    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
    ...                          factor=2, resource='n_estimators',
    ...                          max_resources=30).fit(X, y)
    >>> sh.best_estimator_
    RandomForestClassifier(max_depth=5, n_estimators=24, random_state=0)

لاحظ أنه من غير الممكن الميزانية على معلمة تشكل جزءًا من شبكة المعلمات.

.. _exhausting_the_resources:

استنفاد الموارد المتاحة
 


كما ذكر أعلاه، يعتمد عدد الموارد المستخدمة في كل تكرار على معامل `min_resources`. إذا كان لديك الكثير من الموارد المتاحة ولكنك تبدأ بعدد قليل من الموارد، فقد يتم إهدار بعضها (أي عدم استخدامها)::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.svm import SVC
    >>> from sklearn.experimental import enable_halving_search_cv  # noqa
    >>> from sklearn.model_selection import HalvingGridSearchCV
    >>> import pandas as pd
    >>> param_grid= {'kernel': ('linear', 'rbf'),
    ...              'C': [1, 10, 100]}
    >>> base_estimator = SVC(gamma='scale')
    >>> X, y = make_classification(n_samples=1000)
    >>> sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
    ...                          factor=2, min_resources=20).fit(X, y)
    >>> sh.n_resources_
    [20, 40, 80]

سوف تستخدم عملية البحث 80 موردًا على الأكثر، بينما الحد الأقصى لمقدار الموارد المتاحة لدينا هو ``n_samples=1000``. هنا، لدينا ``min_resources = r_0 = 20``.

بالنسبة إلى :class:`HalvingGridSearchCV`، بشكل افتراضي، يتم تعيين معامل `min_resources` إلى 'exhaust'. هذا يعني أن `min_resources` يتم تعيينه تلقائيًا بحيث يمكن للتكرار الأخير استخدام أكبر عدد ممكن من الموارد، ضمن حد `max_resources`::

    >>> sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
    ...                          factor=2, min_resources='exhaust').fit(X, y)
    >>> sh.n_resources_
    [250, 500, 1000]

تم هنا تعيين `min_resources` تلقائيًا إلى 250، مما يؤدي إلى استخدام التكرار الأخير لجميع الموارد. تعتمد القيمة الدقيقة المستخدمة على عدد معاملات المرشح، وعلى `max_resources` وعلى `factor`.

بالنسبة إلى :class:`HalvingRandomSearchCV`، يمكن استنفاد الموارد بطريقتين:

- عن طريق تعيين `min_resources='exhaust'`، تمامًا مثل :class:`HalvingGridSearchCV`؛
- عن طريق تعيين `n_candidates='exhaust'`.

كلا الخيارين متبادلان بشكل متبادل: استخدام `min_resources='exhaust'` يتطلب معرفة عدد المرشحين، وبالمثل يتطلب `n_candidates='exhaust'` معرفة `min_resources`.

بشكل عام، يؤدي استنفاد إجمالي عدد الموارد إلى مرشح أفضل في النهاية، وهو أكثر استهلاكًا للوقت بشكل طفيف.

.. _aggressive_elimination:

الإزالة العدوانية للمرشحين
------------------------------------

من الناحية المثالية، نريد أن يقيم التكرار الأخير ``factor`` مرشحين (انظر :ref:`amount_of_resource_and_number_of_candidates`). علينا فقط اختيار الأفضل. عندما يكون عدد الموارد المتاحة صغيرًا بالنسبة لعدد المرشحين، فقد يضطر التكرار الأخير إلى تقييم أكثر من ``factor`` مرشحين::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.svm import SVC
    >>> from sklearn.experimental import enable_halving_search_cv  # noqa
    >>> from sklearn.model_selection import HalvingGridSearchCV
    >>> import pandas as pd
    >>>
    >>>
    >>> param_grid = {'kernel': ('linear', 'rbf'),
    ...               'C': [1, 10, 100]}
    >>> base_estimator = SVC(gamma='scale')
    >>> X, y = make_classification(n_samples=1000)
    >>> sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
    ...                          factor=2, max_resources=40,
    ...                          aggressive_elimination=False).fit(X, y)
    >>> sh.n_resources_
    [20, 40]
    >>> sh.n_candidates_
    [6, 3]

نظرًا لأننا لا يمكننا استخدام أكثر من ``max_resources=40`` موارد، يجب أن تتوقف العملية عند التكرار الثاني الذي يقيم أكثر من ``factor=2`` مرشحين.

باستخدام معامل ``aggressive_elimination``، يمكنك إجبار عملية البحث على الانتهاء بأقل من ``factor`` مرشحين في التكرار الأخير. للقيام بذلك، ستقوم العملية بإزالة أكبر عدد ممكن من المرشحين باستخدام ``min_resources`` موارد::

    >>> sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
    ...                            factor=2,
    ...                            max_resources=40,
    ...                            aggressive_elimination=True,
    ...                            ).fit(X, y)
    >>> sh.n_resources_
    [20, 20,  40]
    >>> sh.n_candidates_
    [6, 3, 2]

لاحظ أننا انتهينا مع مرشحين في التكرار الأخير لأننا أزلنا ما يكفي من المرشحين خلال التكرارات الأولى، باستخدام ``n_resources = min_resources = 20``.

.. _successive_halving_cv_results:

تحليل النتائج باستخدام خاصية `cv_results_`

يحتوي سمة "cv_results\_" على معلومات مفيدة لتحليل نتائج البحث. يمكن تحويله إلى إطار بيانات pandasa باستخدام "df = pd.DataFrame(est.cv_results\_)". تشبه سمة "cv_results\_" لـ :class:`HalvingGridSearchCV` و :class:`HalvingRandomSearchCV` تلك الخاصة بـ :class:`GridSearchCV` و :class:`RandomizedSearchCV`، مع معلومات إضافية تتعلق بعملية النصف المتتالي.

فيما يلي مثال يحتوي على بعض أعمدة إطار البيانات (المقتطع):

====  ======  ===============  =================  ========================================================================================
  ..    iter      n_resources    mean_test_score params
====  ======  ===============  =================  ========================================================================================
   0       0              125           0.983667  {'criterion': 'log_loss', 'max_depth': None, 'max_features': 9, 'min_samples_split': 5}
   1       0              125           0.983667  {'criterion': 'gini', 'max_depth': None, 'max_features': 8, 'min_samples_split': 7}
   2       0              125           0.983667  {'criterion': 'gini', 'max_depth': None, 'max_features': 10, 'min_samples_split': 10}
   3       0              125           0.983667  {'criterion': 'log_loss', 'max_depth': None, 'max_features': 6, 'min_samples_split': 6}
 ...     ...              ...                ...  ...
  15       2              500           0.951958  {'criterion': 'log_loss', 'max_depth': None, 'max_features': 9, 'min_samples_split': 10}
  16       2              500           0.947958  {'criterion': 'gini', 'max_depth': None, 'max_features': 10, 'min_samples_split': 10}
  17       2              500           0.951958  {'criterion': 'gini', 'max_depth': None, 'max_features': 10, 'min_samples_split': 4}
  18       3             1000           0.961009  {'criterion': 'log_loss', 'max_depth': None, 'max_features': 9, 'min_samples_split': 10}
  19       3             1000           0.955989  {'criterion': 'gini', 'max_depth': None, 'max_features': 10, 'min_samples_split': 4}
====  ======  ===============  =================  ========================================================================================

كل صف يتوافق مع مجموعة معلمات معينة (مرشح) وتكرار معين. يتم إعطاء التكرار بواسطة عمود "iter". يخبرك عمود "n_resources" بعدد الموارد المستخدمة.

في المثال أعلاه، فإن أفضل مجموعة من المعلمات هي "{'criterion': 'log_loss', 'max_depth': None, 'max_features': 9, 'min_samples_split': 10}" لأنها وصلت إلى التكرار الأخير (3) بأعلى درجة: 0.96.

.. rubric:: المراجع

.. [1] K. Jamieson, A. Talwalkar,
   `Non-stochastic Best Arm Identification and Hyperparameter
   Optimization <http://proceedings.mlr.press/v51/jamieson16.html>`_, in
   proc. of Machine Learning Research, 2

.. [2] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, A. Talwalkar,
   :arxiv:`Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
   <1603.06560>`, in Machine Learning Research 18, 201

.. _grid_search_tips:

نصائح للبحث عن المعلمات
    
(ملاحظة: لم يتم ترجمة النص البرمجي والروابط والتاجات والرموز الخاصة والرموز والمعادلات الرياضية وفقًا للتعليمات).

  تحديد مقياس الهدف
    ------------------
    
    بشكل افتراضي، يستخدم بحث المعلمة دالة "score" الخاصة بالمقدر لتقييم إعداد المعلمة. هذه هي:
    
    * :func:`sklearn.metrics.accuracy_score` للتصنيف
    * :func:`sklearn.metrics.r2_score` للانحدار
    
    بالنسبة لبعض التطبيقات، تكون دالات التقييم الأخرى أكثر ملاءمة (على سبيل المثال في التصنيف غير المتوازن، غالبًا ما تكون دالة الدقة غير مفيدة). يمكن تحديد دالة تقييم بديلة عبر معلمة "scoring" لمعظم أدوات بحث المعلمة. راجع :ref:`scoring_parameter` للحصول على مزيد من التفاصيل.
    
    تحديد مقاييس متعددة للتقييم
    ----------------------------------
    
    تسمح :class:`GridSearchCV` و :class:`RandomizedSearchCV` بتحديد مقاييس متعددة لمعلمة "scoring".
    
    يمكن تحديد التقييم متعدد المقاييس إما كقائمة من سلاسل أسماء المقاييس المعرفة مسبقًا أو قاموس يرسم اسم الدالة المسجلة إلى دالة الدالة المسجلة و/أو اسم (أسماء) الدالة المسجلة المعرفة مسبقًا. راجع :ref:`multimetric_scoring` للحصول على مزيد من التفاصيل.
    
    عند تحديد مقاييس متعددة، يجب تعيين معلمة "refit" إلى المقياس (السلسلة) الذي سيتم العثور على "best_params_" له واستخدامه لبناء "best_estimator_" على مجموعة البيانات بأكملها. إذا لم يكن من المفترض إعادة التجهيز، فاضبط "refit=False". سيؤدي ترك refit إلى القيمة الافتراضية "None" إلى حدوث خطأ عند استخدام مقاييس متعددة.
    
    راجع :ref:`sphx_glr_auto_examples_model_selection_plot_multi_metric_evaluation.py` للحصول على مثال للاستخدام.
    
    لا تدعم :class:`HalvingRandomSearchCV` و :class:`HalvingGridSearchCV` التقييم متعدد المقاييس.
    
    المقدرون المركبون ومساحات المعلمة
    ----------------------------------
    
    تسمح :class:`GridSearchCV` و :class:`RandomizedSearchCV` بالبحث عبر معلمات المقدرين المركبين أو المتداخلين مثل:
    
    * :class:`~sklearn.pipeline.Pipeline`
    * :class:`~sklearn.compose.ColumnTransformer`
    * :class:`~sklearn.ensemble.VotingClassifier`
    * :class:`~sklearn.calibration.CalibratedClassifierCV`
    
    باستخدام بناء جملة مخصص ``<estimator>__<parameter>``::
    
      >>> from sklearn.model_selection import GridSearchCV
      >>> from sklearn.calibration import CalibratedClassifierCV
      >>> from sklearn.ensemble import RandomForestClassifier
      >>> from sklearn.datasets import make_moons
      >>> X, y = make_moons()
      >>> calibrated_forest = CalibratedClassifierCV(
      ...    estimator=RandomForestClassifier(n_estimators=10))
      >>> param_grid = {
      ...    'estimator__max_depth': [2, 4, 6, 8]}
      >>> search = GridSearchCV(calibrated_forest, param_grid, cv=5)
      >>> search.fit(X, y)
      GridSearchCV(cv=5,
                   estimator=CalibratedClassifierCV(...),
                   param_grid={'estimator__max_depth': [2, 4, 6, 8]})
    
    هنا، ``<estimator>`` هو اسم معلمة المقدّر المتداخل، في هذه الحالة ``estimator``.
    إذا تم إنشاء المقدّر الفوقي كمجموعة من المقدّرين كما هو الحال في `pipeline.Pipeline`، فإن ``<estimator>`` يشير إلى اسم المقدّر، راجع :ref:`pipeline_nested_parameters`. في الممارسة العملية، يمكن أن يكون هناك عدة مستويات من التداخل::
    
      >>> from sklearn.pipeline import Pipeline
      >>> from sklearn.feature_selection import SelectKBest
      >>> pipe = Pipeline([
      ...    ('select', SelectKBest()),
      ...    ('model', calibrated_forest)])
      >>> param_grid = {
      ...    'select__k': [1, 2],
      ...    'model__estimator__max_depth': [2, 4, 6, 8]}
      >>> search = GridSearchCV(pipe, param_grid, cv=5).fit(X, y)
    
    يرجى الرجوع إلى :ref:`pipeline` لإجراء عمليات بحث عن المعلمة عبر خطوط الأنابيب.
    
    اختيار النموذج: التطوير والتقييم
    ----------------------------------
    
    يمكن اعتبار اختيار النموذج من خلال تقييم إعدادات المعلمة المختلفة كطريقة لاستخدام البيانات المسمىّة لـ "تدريب" معلمات الشبكة.
    
    عند تقييم النموذج الناتج، من المهم القيام بذلك على العينات المحجوزة التي لم يتم رؤيتها أثناء عملية البحث في الشبكة:
    من المستحسن تقسيم البيانات إلى **مجموعة تطوير** (لتغذيتها إلى مثيل :class:`GridSearchCV`) و **مجموعة تقييم**
    لحساب مقاييس الأداء.
    
    يمكن القيام بذلك باستخدام وظيفة الأداة المساعدة :func:`train_test_split`.
    
    التوازي
    --------
    
    تقوم أدوات بحث المعلمة بتقييم كل مجموعة من المعلمات بشكل مستقل على كل طية بيانات. يمكن تشغيل العمليات بالتوازي باستخدام الكلمة الأساسية
    ``n_jobs=-1``. راجع توقيع الدالة لمزيد من التفاصيل، وكذلك مدخل المسرد لـ :term:`n_jobs`.
    
    المتانة للفشل
    ----------------
    
    قد تؤدي بعض إعدادات المعلمة إلى فشل ``fit`` لواحدة أو أكثر من طيات البيانات.
    بشكل افتراضي، سيؤدي هذا إلى فشل البحث بالكامل، حتى إذا
    يمكن تقييم بعض إعدادات المعلمة بالكامل. سيؤدي تعيين ``error_score=0``
    (أو `=np.nan`) إلى جعل الإجراء متينًا لمثل هذا الفشل، وإصدار تحذير وتعيين الدرجة لتلك الطية على 0 (أو `nan`)، ولكن إكمال
    البحث.
    
    .. _alternative_cv:
    
    بدائل لبحث المعلمة بالقوة الغاشمة
    
    (لا توجد ترجمة لهذه الفقرة لأنها لا تحتوي على أي نص يجب ترجمته.)
    
    (No translation for this paragraph as it does not contain any text to be translated.)

============================================

التحقق من صحة النموذج المحدد
-------------------------------

يمكن لبعض النماذج أن تتناسب مع البيانات لمجموعة من القيم لبعض المعلمات بكفاءة تقارب كفاءة تقدير المقدر لقيمة واحدة من المعلمة. يمكن الاستفادة من هذه الميزة لإجراء تحقق أكثر كفاءة من الصحة يستخدم لاختيار النموذج لهذه المعلمة.

المعلمة الأكثر شيوعًا التي يمكن تطبيق هذه الاستراتيجية عليها هي المعلمة التي ترمز إلى قوة منظم. في هذه الحالة ، نقول أننا نحسب **مسار التنظيم** للمقدر.

فيما يلي قائمة بهذه النماذج:

.. currentmodule:: sklearn

.. autosummary::

   linear_model.ElasticNetCV
   linear_model.LarsCV
   linear_model.LassoCV
   linear_model.LassoLarsCV
   linear_model.LogisticRegressionCV
   linear_model.MultiTaskElasticNetCV
   linear_model.MultiTaskLassoCV
   linear_model.OrthogonalMatchingPursuitCV
   linear_model.RidgeCV
   linear_model.RidgeClassifierCV


معيار المعلومات
---------------------

يمكن لبعض النماذج أن تقدم صيغة مغلقة نظرية معلوماتية للتقدير الأمثل لمعلمة التنظيم من خلال حساب مسار تنظيم واحد (بدلاً من عدة مسارات عند استخدام التحقق من الصحة المتقاطع).

فيما يلي قائمة النماذج التي تستفيد من معيار معلومات آكايك (AIC) أو معيار معلومات بايز (BIC) لاختيار النموذج الآلي:

.. autosummary::

   linear_model.LassoLarsIC


.. _out_of_bag:

تقديرات خارج الحقيبة
--------------------

عند استخدام طرق مجموعة تعتمد على الحقيبة ، أي توليد مجموعات بيانات تدريب جديدة باستخدام أخذ العينات مع الاستبدال ، يظل جزء من مجموعة التدريب غير مستخدم. لكل مصنف في المجموعة ، يتم ترك جزء مختلف من مجموعة التدريب.

يمكن استخدام هذا الجزء المتبقي لتقدير خطأ التعميم دون الحاجة إلى الاعتماد على مجموعة تحقق منفصلة. يأتي هذا التقدير "مجانًا" حيث لا توجد حاجة إلى بيانات إضافية ويمكن استخدامه لاختيار النموذج.

يتم تنفيذ هذا حاليًا في الفئات التالية:

.. autosummary::

    ensemble.RandomForestClassifier
    ensemble.RandomForestRegressor
    ensemble.ExtraTreesClassifier
    ensemble.ExtraTreesRegressor
    ensemble.GradientBoostingClassifier
    ensemble.GradientBoostingRegressor
    

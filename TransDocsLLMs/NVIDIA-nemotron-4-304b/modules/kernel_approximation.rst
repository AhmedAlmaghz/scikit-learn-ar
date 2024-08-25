.. _kernel_approximation:

تقريب النواة (Kernel Approximation)
====================

    يحتوي هذا النمط الفرعي على وظائف تقريبية لرسم الخرائط المميزة التي تتوافق مع نواة معينة، كما يتم استخدامها على سبيل المثال في آلات المتجهات الداعمة (انظر: ref: `svm`).
    تقوم وظائف الميزة التالية بإجراء تحويلات غير خطية للمدخلات، والتي يمكن أن تكون بمثابة أساس للتصنيف الخطي أو الخوارزميات الأخرى.

    .. currentmodule:: sklearn.linear_model

    إن ميزة استخدام خرائط الميزات الصريحة التقريبية مقارنة بـ "خدعة النواة" (<https://en.wikipedia.org/wiki/Kernel_trick>)، والتي تستخدم خرائط الميزات بشكل ضمني، هي أن الخرائط الصريحة يمكن أن تكون أكثر ملاءمة للتعلم عبر الإنترنت ويمكن أن تقلل بشكل كبير من تكلفة التعلم مع مجموعات البيانات الكبيرة جدًا.
    لا تتدرج آلات المتجهات الداعمة القياسية جيدًا لمجموعات البيانات الكبيرة، ولكن باستخدام خريطة نواة تقريبية، من الممكن استخدام آلات المتجهات الداعمة الخطية الأكثر كفاءة.
    على وجه الخصوص، يمكن أن يجعل الجمع بين تقريبات خريطة النواة و: class: `SGDClassifier` التعلم غير الخطي على مجموعات البيانات الكبيرة ممكنًا.

    نظرًا لعدم وجود الكثير من العمل التجريبي باستخدام عمليات التضمين التقريبية، فمن المستحسن مقارنة النتائج مع طرق النواة الدقيقة عند الإمكان.

    .. seealso::

       :ref:`polynomial_regression` لتحويل متعدد الحدود دقيق.

    .. currentmodule:: sklearn.kernel_approximation

    .. _nystroem_kernel_approx:

    طريقة نيستروم لتقريب النواة
    ----------------------------------------
    طريقة نيستروم، كما هي مطبقة في: class: `Nystroem` هي طريقة عامة لتقريب النواة من الرتبة المنخفضة. تحقق ذلك عن طريق أخذ عينات دون استبدال صفوف / أعمدة البيانات التي يتم تقييم النواة عليها. في حين أن التعقيد الحسابي للطريقة الدقيقة هو: math: `mathcal {O} (n ^ 3 _ {text {samples}}) `، فإن تعقيد التقريب هو: math: `mathcal {O} (n ^ 2 _ {text {components}} cdot n _ {text {samples}}) `، حيث يمكن للمرء أن يحدد: math: `n _ {text {components}} ll n _ {text {samples}} `دون انخفاض كبير في الأداء [WS200

    يمكننا بناء تحليل القيمة الذاتية لمصفوفة النواة: math: `K`، بناءً على ميزات البيانات، ثم تقسيمها إلى نقاط بيانات تم أخذ عينات منها ولم يتم أخذ عينات منها.

    .. math::

            K = U Λ U ^ T
            = begin {bmatrix} U_1 \\\\ U_2 end {bmatrix} Λ begin {bmatrix} U_1 \\\\ U_2 end {bmatrix} ^ T
            = begin {bmatrix} U_1 Λ U_1 ^ T & U_1 Λ U_2 ^ T \\\\ U_2 Λ U_1 ^ T & U_2 Λ U_2 ^ T end {bmatrix}
            equiv begin {bmatrix} K_ {11} & K_ {12} \\\\ K_ {21} & K_ {22} end {bmatrix}

    حيث:

    * :math:`U` هي مصفوفة متعامدة
    * :math:`Λ` هي مصفوفة قطرية من القيم الذاتية
    * :math:`U_1` هي مصفوفة متعامدة من العينات التي تم اختيارها
    * :math:`U_2` هي مصفوفة متعامدة من العينات التي لم يتم اختيارها

    بالنظر إلى أنه يمكن الحصول على: math:`U_1 Λ U_1 ^ T` عن طريق تطبيع مصفوفة: math:`K_ {11}`, ويمكن تقييم: math:`U_2 Λ U_1 ^ T` (وكذلك عبرها)، فإن المصطلح المتبقي الوحيد الذي يجب توضيحه هو: math:`U_2 Λ U_2 ^ T`. للقيام بذلك، يمكننا التعبير عنها من حيث المصفوفات التي تم تقييمها بالفعل:

    .. math::

             begin {align} U_2 Λ U_2 ^ T & = left (K_ {21} U_1 Λ ^ {-1} right) Λ left (K_ {21} U_1 Λ ^ {-1} right) ^ T
             \\\ & = K_ {21} U_1 (Λ ^ {-1} Λ) Λ ^ {-1} U_1 ^ T K_ {21} ^ T
             \\\ & = K_ {21} U_1 Λ ^ {-1} U_1 ^ T K_ {21} ^ T
             \\\ & = K_ {21} K_ {11} ^ {-1} K_ {21} ^ T
             \\\ & = left (K_ {21} K_ {11} ^ {- frac12} right) left (K_ {21} K_ {11} ^ {- frac12} right) ^ T
             . end {align}

    أثناء ``fit``، تقوم الفئة: class: `Nystroem` بتقييم الأساس: math:`U_1`, وتحسب ثابت التطبيع،: math:`K_ {11} ^ {- frac12}`. لاحقًا، أثناء ``transform``، يتم تحديد مصفوفة النواة بين الأساس (الذي يعطيه السمة `components_`) ونقاط البيانات الجديدة، ``X``. ثم يتم ضرب هذه المصفوفة في مصفوفة ``normalization_`` للنتيجة النهائية.

    تستخدم الفئة: class: `Nystroem` بشكل افتراضي نواة ``rbf``، ولكن يمكنها استخدام أي وظيفة نواة أو مصفوفة نواة محوسبة مسبقًا. يتم إعطاء عدد العينات المستخدمة - وهو أيضًا أبعاد الميزات المحسوبة - بواسطة المعلمة ``n_components``.

    .. rubric :: أمثلة

    * انظر المثال بعنوان: ref: `sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`، الذي يوضح خط أنابيب تعلم آلي فعال يستخدم نواة: class: `Nystroem`.

    .. _rbf_kernel_approx:

    نواة دالة الأساس الشعاعي
  ----------------------------

يقوم :class:`RBFSampler` بإنشاء خريطة تقريبية لوظيفة الأساس الشعاعي، والمعروفة أيضًا باسم "Random Kitchen Sinks" [RR2007]_. يمكن استخدام هذا التحويل لنمذجة خريطة النواة بشكل صريح، قبل تطبيق خوارزمية خطية، على سبيل المثال SVM خطي::

    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSampler(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=5)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=5)
    >>> clf.score(X_features, y)
    1.0

يعتمد التعيين على تقريب مونت كارلو لقيم النواة. تقوم دالة "fit" بإجراء أخذ عينات مونت كارلو، بينما تقوم طريقة "transform" بتنفيذ تعيين البيانات. نظرًا للطبيعة العشوائية للعملية، قد تختلف النتائج بين المكالمات المختلفة لوظيفة "fit".

تأخذ دالة "fit" وسيطين: "n_components"، وهو البعد المستهدف لتحويل الميزة، و "gamma"، معامل نواة RBF. سيؤدي "n_components" الأعلى إلى تقريب أفضل للنواة وسيؤدي إلى نتائج أكثر تشابهًا مع تلك التي تنتجها SVM النواة. لاحظ أن "تثبيت" دالة الميزة لا يعتمد في الواقع على البيانات المقدمة إلى دالة "fit". يتم استخدام أبعاد البيانات فقط. يمكن العثور على التفاصيل حول الطريقة في [RR2007]_.

بالنسبة لقيمة معينة من "n_components"، غالبًا ما يكون :class:`RBFSampler` أقل دقة من :class:`Nystroem`. ومع ذلك، فإن :class:`RBFSampler` أرخص في الحساب، مما يجعل استخدام مساحات الميزات الأكبر أكثر كفاءة.

.. figure:: ../auto_examples/miscellaneous/images/sphx_glr_plot_kernel_approximation_002.png
    :target: ../auto_examples/miscellaneous/plot_kernel_approximation.html
    :scale: 50%
    :align: center

    مقارنة نواة RBF الدقيقة (يسار) مع التقريب (يمين)

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_miscellaneous_plot_kernel_approximation.py`

.. _additive_chi_kernel_approx:

نواة Chi المضافة
---------------------------

نواة Chi المضافة هي نواة على التوزيعات التكرارية، وغالبًا ما تستخدم في رؤية الكمبيوتر.

يتم إعطاء نواة Chi المضافة كما هو مستخدم هنا بواسطة

.. math::

        k(x, y) = \sum_i \frac{2x_iy_i}{x_i+y_i}

هذا ليس هو نفسه تمامًا :func:`sklearn.metrics.pairwise.additive_chi2_kernel`. يفضل مؤلفو [VZ2010]_ الإصدار أعلاه لأنه دائمًا موجب محدد.
نظرًا لأن النواة مضافة، فمن الممكن معالجة جميع المكونات :math:`x_i` بشكل منفصل للتضمين. هذا يجعل من الممكن أخذ عينات من تحويل فورييه في فترات منتظمة، بدلاً من التقريب باستخدام أخذ عينات مونت كارلو.

تنفذ فئة :class:`AdditiveChi2Sampler` هذا أخذ العينات الحتمي لكل مكون. يتم أخذ عينات من كل مكون :math:`n` مرات، مما ينتج عنه :math:`2n+1` أبعاد لكل بُعد إدخال (يأتي المضاعف من الجزء الحقيقي والخيالي لتحويل فورييه).
في الأدبيات، عادة ما يتم اختيار :math:`n` ليكون 1 أو 2، مما يحول مجموعة البيانات إلى حجم ``n_samples * 5 * n_features`` (في حالة :math:`n=2`).

يمكن دمج خريطة الميزة التقريبية التي يوفرها :class:`AdditiveChi2Sampler` مع خريطة الميزة التقريبية التي يوفرها :class:`RBFSampler` لإنتاج خريطة ميزة تقريبية لنواة Chi المربعة الأسية.
راجع [VZ2010]_ للتفاصيل و [VVZ2010]_ للجمع مع :class:`RBFSampler`.

.. _skewed_chi_kernel_approx:

نواة Chi المائلة
-------------------------

يتم إعطاء نواة Chi المائلة بواسطة:

.. math::

        k(x,y) = \prod_i \frac{2\sqrt{x_i+c}\sqrt{y_i+c}}{x_i + y_i + 2c}


لديها خصائص مشابهة لنواة Chi المربعة الأسية المستخدمة غالبًا في رؤية الكمبيوتر، ولكنها تسمح بتقريب بسيط لميزة مونت كارلو.

استخدام :class:`SkewedChi2Sampler` هو نفسه الاستخدام الموصوف أعلاه لـ :class:`RBFSampler`. الفرق الوحيد هو في المعلمة الحرة، التي تسمى :math:`c`.
للحصول على الدافع لهذا التعيين والتفاصيل الرياضية، انظر [LS201

...

???????

.. _polynomial_kernel_approx:

تقريب نواة متعدد الحدود عبر Tensor Sketch
    ---

    إن نواة متعدد الحدود (:ref:`polynomial_kernel`) هي نوع شائع من دالات النواة وتعطى بالمعادلة:

    .. math::

            k(x, y) = (\gamma x^\top y +c_0)^d

    حيث:

    * ``x``، ``y`` هما متجها الدخل
    * ``d`` هي درجة النواة

    بشكل بديهي، تتكون مساحة الميزة لنواة متعدد الحدود من الدرجة `d` من جميع منتجات الدرجة-`d` الممكنة بين ميزات الدخل، مما يسمح لخوارزميات التعلم التي تستخدم هذه النواة أن تأخذ في الاعتبار التفاعلات بين الميزات.

    طريقة TensorSketch [PP2013]_، كما هي مطبقة في :class:`PolynomialCountSketch`، هي طريقة قابلة للتطوير ومستقلة عن بيانات الدخل لتقريب نواة متعدد الحدود. وهي تعتمد على مفهوم Count sketch [WIKICS]_ [CCF2002]_، وهي تقنية تقليل الأبعاد مشابهة لـ feature hashing، والتي تستخدم بدلاً من ذلك عدة دالات تجزئة مستقلة. تحصل TensorSketch على Count Sketch للمنتج الخارجي لمتجهين (أو متجه مع نفسه)، والذي يمكن استخدامه كتقريب لمساحة ميزة نواة متعدد الحدود. على وجه الخصوص، بدلاً من حساب المنتج الخارجي بشكل صريح، تحسب TensorSketch Count Sketch للمتجهات ثم تستخدم الضرب متعدد الحدود عبر تحويل فورييه السريع لحساب Count Sketch للمنتج الخارجي الخاص بها.

    بشكل ملائم، تتكون مرحلة التدريب في TensorSketch ببساطة من تهيئة بعض المتغيرات العشوائية. وبالتالي فهي مستقلة عن بيانات الدخل، أي أنها تعتمد فقط على عدد ميزات الدخل، وليس قيم البيانات. بالإضافة إلى ذلك، يمكن لهذه الطريقة تحويل العينات في :math:`\mathcal{O}(n_{\text{samples}}(n_{\text{features}} + n_{\text{components}} \log(n_{\text{components}})))` زمن، حيث :math:`n_{\text{components}}` هو البعد المخرج المرغوب، والذي يحدده ``n_components``.

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_kernel_approximation_plot_scalable_poly_kernels.py`

    .. _tensor_sketch_kernel_approx:

    التفاصيل الرياضية
    --------------------

    تعتمد طرق النواة مثل آلات الدعم المتجه أو PCA المعتمدة على النواة على خاصية فضاءات هيلبرت المتناسخة. لأي دالة نواة موجبة محددة :math:`k` (ما يسمى نواة ميرسر)، من المضمون وجود تعيين :math:`\phi` إلى فضاء هيلبرت :math:`\mathcal{H}`، بحيث

    .. math::

            k(x,y) = \langle \phi(x), \phi(y) \rangle

    حيث :math:`\langle \cdot, \cdot \rangle` تدل على المنتج الداخلي في فضاء هيلبرت.

    إذا كان أحد الخوارزميات، مثل آلة الدعم المتجه الخطي أو PCA، يعتمد فقط على المنتج القياسي لنقاط البيانات :math:`x_i`، فيمكن للمرء استخدام قيمة :math:`k(x_i, x_j)`, والتي تتوافق مع تطبيق الخوارزمية على نقاط البيانات المرسومة :math:`\phi(x_i)`. ميزة استخدام :math:`k` هي أنه لا يتعين حساب التعيين :math:`\phi` صراحةً، مما يسمح بميزات تعسفية كبيرة (حتى لانهائية).

    أحد عيوب طرق النواة هو أنه قد يكون من الضروري تخزين العديد من قيم النواة :math:`k(x_i, x_j)` أثناء التحسين. إذا تم تطبيق مصنف معتمد على النواة على بيانات جديدة :math:`y_j`، فيجب حساب :math:`k(x_i, y_j)` لإجراء تنبؤات، ربما للعديد من :math:`x_i` المختلفة في مجموعة التدريب.

    تسمح الفئات الموجودة في هذه الوحدة الفرعية بتقريب التضمين :math:`\phi`، وبالتالي العمل صراحةً مع التمثيلات :math:`\phi(x_i)`, مما يلغي الحاجة إلى تطبيق النواة أو تخزين أمثلة التدريب.

    .. rubric:: المراجع

    .. [WS2001] `"Using the Nyström method to speed up kernel machines"
      <https://papers.nips.cc/paper_files/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html>`_
      Williams, C.K.I.; Seeger, M. - 2001.
    .. [RR2007] `"Random features for large-scale kernel machines"
      <https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html>`_
      Rahimi, A. and Recht, B. - Advances in neural information processing 2007,
    .. [LS2010] `"Random Fourier approximations for skewed multiplicative histogram kernels"
      <https://www.researchgate.net/publication/221114584_Random_Fourier_Approximations_for_Skewed_Multiplicative_Histogram_Kernels>`_
      Li, F., Ionescu, C., and Sminchisescu, C.
      - Pattern Recognition,  DAGM 2010, Lecture Notes in Computer Science.
    .. [VZ2010] `"Efficient additive kernels via explicit feature maps"
      <https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf>`_
      Vedaldi, A. and Zisserman, A. - Computer Vision and Pattern Recognition 2010
    .. [VVZ2010] `"Generalized RBF feature maps for Efficient Detection"
      <https://www.robots.ox.ac.uk/~vgg/publications/2010/Sreekanth10/sreekanth10.pdf>`_
      Vempati, S. and Vedaldi, A. and Zisserman, A. and Jawahar, CV - 2010
    .. [PP2013] :doi:`"Fast and scalable polynomial kernels via explicit feature maps"
      <10.1145/2487575.2487591>`
      Pham, N., & Pagh, R. - 2013
    .. [CCF2002] `"Finding frequent items in data streams"
      <https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarCF.pdf>`_
      Charikar, M., Chen, K., & Farach-Colton - 2002
    .. [WIKICS] `"Wikipedia: Count sketch"
      <https://en.wikipedia.org/wiki/Count_sketch>`_
    
    ---

.. currentmodule:: sklearn.manifold

 .. _manifold:

 =================
 تعلم المنوعات
 =================

 .. rst-class:: quote

                  | ابحث عن الضروريات العارية
                  | الوصفات البسيطة العارية
                  | انسى همومك وصراعاتك
                  | أعني الضروريات العارية
                  | وصفات الطبيعة الأم القديمة
                  | التي تجلب الضروريات العارية للحياة
                  |
                  |             -- أغنية Baloo [The Jungle Book]



 .. figure:: ../auto_examples/manifold/images/sphx_glr_plot_compare_methods_001.png
    :target: ../auto_examples/manifold/plot_compare_methods.html
    :align: center
    :scale: 70%

 .. |manifold_img3| image:: ../auto_examples/manifold/images/sphx_glr_plot_compare_methods_003.png
    :target: ../auto_examples/manifold/plot_compare_methods.html
    :scale: 60%

 .. |manifold_img4| image:: ../auto_examples/manifold/images/sphx_glr_plot_compare_methods_004.png
       :target: ../auto_examples/manifold/plot_compare_methods.html
       :scale: 60%

 .. |manifold_img5| image:: ../auto_examples/manifold/images/sphx_glr_plot_compare_methods_005.png
       :target: ../auto_examples/manifold/plot_compare_methods.html
       :scale: 60%

 .. |manifold_img6| image:: ../auto_examples/manifold/images/sphx_glr_plot_compare_methods_006.png
       :target: ../auto_examples/manifold/plot_compare_methods.html
       :scale: 60%

 .. centered:: |manifold_img3| |manifold_img4| |manifold_img5| |manifold_img6|


 تعلم المنوعات هو نهج للحد من الأبعاد غير الخطية.
 تعتمد الخوارزميات لهذه المهمة على فكرة أن أبعاد
 العديد من مجموعات البيانات مرتفعة بشكل مصطنع فقط.


 مقدمة
 ========

 يمكن أن تكون مجموعات البيانات عالية الأبعاد صعبة للغاية للتصور.  في حين أن البيانات
 في بعدين أو ثلاثة أبعاد يمكن رسمها لإظهار الهيكل
 المتأصل في البيانات، فإن الرسوم البيانية عالية الأبعاد المكافئة أقل
 بديهية.  لمساعدة تصور بنية مجموعة بيانات، يجب
 تقليل البعد بطريقة ما.

 أبسط طريقة لتحقيق هذا التخفيض في الأبعاد هي عن طريق أخذ
 إسقاط عشوائي للبيانات.  على الرغم من أن هذا يسمح ببعض
 درجة تصور بنية البيانات، فإن عشوائية الاختيار تترك الكثير
 مرغوب فيه.  في إسقاط عشوائي، من المحتمل أن يكون أكثر
 سيتم فقدان بنية مثيرة للاهتمام داخل البيانات.


 .. |digits_img| image:: ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_001.png
       :target: ../auto_examples/manifold/plot_lle_digits.html
       :scale: 50

 .. |projected_img| image::  ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_002.png
       :target: ../auto_examples/manifold/plot_lle_digits.html
       :scale: 50

 .. centered:: |digits_img| |projected_img|


 لمعالجة هذا القلق، تم تصميم عدد من الأطر الخطية الخاضعة للإشراف وغير الخاضعة للإشراف
 للحد من الأبعاد، مثل تحليل المكونات الرئيسية (PCA)، وتحليل المكونات المستقلة، والتحليل التمييزي الخطي، وغيرها.  تحدد هذه الخوارزميات معايير محددة
 لاختيار إسقاط "مثير للاهتمام" خطي للبيانات.
 يمكن أن تكون هذه الأساليب قوية، ولكنها غالبًا ما تفوت بنية مهمة غير خطية
 في البيانات.


 .. |PCA_img| image:: ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_003.png
       :target: ../auto_examples/manifold/plot_lle_digits.html
       :scale: 50

 .. |LDA_img| image::  ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_004.png
       :target: ../auto_examples/manifold/plot_lle_digits.html
       :scale: 50

 .. centered:: |PCA_img| |LDA_img|

 يمكن اعتبار تعلم المنوعات كمحاولة لتعميم الخطي
 أطر مثل PCA لتكون حساسة للهيكل غير الخطي في البيانات. على الرغم من أن
 توجد متغيرات خاضعة للإشراف، فإن مشكلة تعلم المنوعات النموذجية هي
 غير خاضعة للإشراف: فهي تتعلم البنية عالية الأبعاد للبيانات
 من البيانات نفسها، دون استخدام التصنيفات المحددة مسبقًا.


 .. rubric:: أمثلة

 * انظر :ref:`sphx_glr_auto_examples_manifold_plot_lle_digits.py` للحصول على مثال عن
   الحد من الأبعاد على الأرقام المكتوبة بخط اليد.

 * انظر :ref:`sphx_glr_auto_examples_manifold_plot_compare_methods.py` للحصول على مثال عن
   الحد من الأبعاد على مجموعة بيانات "S-curve" لعبة.

 يتم تلخيص عمليات تنفيذ تعلم المنوعات المتاحة في scikit-learn أدناه

 .. _isomap:

 Isomap
    
 
   Isomap seeks to preserve the intrinsic geometry of the data, and
   thus the geodesic distances between all points.  It does this by
   approximating the geodesic distances between all points using a
   graph.  The graph can be thought of as a discrete approximation of
   the low-dimensional manifold in high-dimensional space.  The
   shortest path between two points on the graph is then an
   approximation of the geodesic distance between the points on the
   manifold.  The low-dimensional embedding is found which preserves
   these pairwise distances as well as possible.

   Isomap is implemented in :class:`sklearn.manifold.Isomap`.


.. _locally_linear_embedding:

Locally Linear Embedding
    
 
   Locally Linear Embedding (LLE) reduces the dimensionality of data
   while preserving the geometry of the local neighborhoods of the
   points.  It does this by finding a low-dimensional projection of
   the data in which the relationship between a point and its
   neighbors is maintained as well as possible.

   LLE is implemented in :class:`sklearn.manifold.LocallyLinearEmbedding`.


.. _hessian_lle:

Hessian Eigenmaps
    
 
   Hessian Eigenmaps (HLLE) is a further extension of LLE which
   attempts to maintain the local curvature of the high-dimensional
   data in the low-dimensional projection.  It does this by
   maintaining the Hessian of the data at each point, which can be
   thought of as a higher-order analogue of the Laplacian used in
   LLE.

   HLLE is implemented in :class:`sklearn.manifold.HessianLLE`.


.. _modified_neighbors_embedding:

Modified Neighborhood Embedding
    
 
   Modified Neighborhood Embedding (MNE) is a variant of LLE which
   seeks to maintain the local reconstruction weights, but uses a
   different cost function for the optimization.  This leads to a
   closed-form solution for the low-dimensional embedding, which can
   be solved more efficiently than the other LLE variants.

   MNE is implemented in :class:`sklearn.manifold.ModifiedNeighborEmbedding`.


.. _t_sne:

t-distributed Stochastic Neighbor Embedding
    
 
   t-distributed Stochastic Neighbor Embedding (t-SNE) minimizes the
   divergence between two distributions: a distribution that measures
   pairwise similarities of the input objects, and a distribution that
   measures pairwise similarities of the corresponding low-dimensional
   points in the embedding.

   t-SNE is implemented in :class:`sklearn.manifold.TSNE`.


.. _spectral_embedding:

Spectral Embedding
    
 
   Spectral Embedding (Laplacian Eigenmaps) is a manifold learning
   technique that provides a low-dimensional representation of the
   data using
إحدى أولى الطرق المستخدمة في تعلم المنوعات هي خوارزمية Isomap، وهي اختصار لـ Isometric Mapping. يمكن اعتبار Isomap امتدادًا لـ Multi-dimensional Scaling (MDS) أو Kernel PCA. تسعى Isomap إلى إيجاد تضمين ذي أبعاد أقل يحافظ على المسافات الجيوديسية بين جميع النقاط. يمكن تنفيذ Isomap باستخدام الكائن :class:`Isomap`.

.. figure:: ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_005.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

.. dropdown:: التعقيد

  تتكون خوارزمية Isomap من ثلاث مراحل:

  1. **البحث عن الجيران الأقرب.** تستخدم Isomap :class:`~sklearn.neighbors.BallTree` للبحث الفعال عن الجيران. التكلفة تقريبية :math:`O[D \log(k) N \log(N)]`، لـ :math:`k` من الجيران الأقرب لـ :math:`N` نقطة في :math:`D` أبعاد.

  2. **البحث في الرسم البياني لأقصر مسار.** أكثر الخوارزميات فعالية لهذا هي *خوارزمية Dijkstra*، والتي تقريبية :math:`O[N^2(k + \log(N))]`، أو *خوارزمية Floyd-Warshall*، والتي هي :math:`O[N^3]`. يمكن للمستخدم اختيار الخوارزمية باستخدام الكلمة المفتاحية ``path_method`` لـ ``Isomap``. إذا لم يتم تحديدها، فإن الكود يحاول اختيار أفضل خوارزمية لبيانات الإدخال.

  3. **تحلل القيم الذاتية الجزئي.** يتم ترميز التضمين في متجهات القيم الذاتية المقابلة لـ :math:`d` أكبر القيم الذاتية لنواة Isomap :math:`N \times N`. للتحلل الكثيف، التكلفة تقريبية :math:`O[d N^2]`. يمكن غالبًا تحسين هذه التكلفة باستخدام محلل ``ARPACK``. يمكن للمستخدم تحديد محلل القيم الذاتية باستخدام الكلمة المفتاحية ``eigen_solver`` لـ ``Isomap``. إذا لم يتم تحديدها، فإن الكود يحاول اختيار أفضل خوارزمية لبيانات الإدخال.

  التعقيد الكلي لـ Isomap هو :math:`O[D \log(k) N \log(N)] + O[N^2(k + \log(N))] + O[d N^2]`.

  * :math:`N` : عدد نقاط بيانات التدريب
  * :math:`D` : البعد المدخلي
  * :math:`k` : عدد الجيران الأقرب
  * :math:`d` : البعد المخرجي

.. rubric:: المراجع

* `"A global geometric framework for nonlinear dimensionality reduction"
  <http://science.sciencemag.org/content/290/5500/2319.full>`_
  Tenenbaum, J.B.; De Silva, V.; & Langford, J.C.  Science 290 (5500)

.. _locally_linear_embedding:

التضمين الخطي المحلي
========================

يسعى التضمين الخطي المحلي (LLE) إلى إيجاد إسقاط ذي أبعاد أقل للبيانات يحافظ على المسافات داخل الأحياء المحلية. يمكن اعتباره سلسلة من تحليلات المكون الرئيسي المحلي التي تتم مقارنتها عالميًا لإيجاد أفضل تضمين غير خطي.

يمكن تنفيذ التضمين الخطي المحلي باستخدام الدالة :func:`locally_linear_embedding` أو نظيرتها الموجهة للكائنات :class:`LocallyLinearEmbedding`.

.. figure:: ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_006.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

.. dropdown:: التعقيد

  تتكون خوارزمية LLE القياسية من ثلاث مراحل:

  1. **البحث عن الجيران الأقرب.** انظر المناقشة تحت Isomap أعلاه.

  2. **بناء مصفوفة الوزن.** :math:`O[D N k^3]`. يتضمن بناء مصفوفة وزن LLE حل معادلة خطية :math:`k \times k` لكل من الأحياء المحلية :math:`N`.

  3. **تحلل القيم الذاتية الجزئي.** انظر المناقشة تحت Isomap أعلاه.

  التعقيد الكلي لـ LLE القياسي هو :math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[d N^2]`.

  * :math:`N` : عدد نقاط بيانات التدريب
  * :math:`D` : البعد المدخلي
  * :math:`k` : عدد الجيران الأقرب
  * :math:`d` : البعد المخرجي

.. rubric:: المراجع

* `"Nonlinear dimensionality reduction by locally linear embedding"
  <http://www.sciencemag.org/content/290/5500/2323.full>`_
  Roweis, S. & Saul, L.  Science 290:2323 (2000)


التضمين الخطي المحلي المعدل

إحدى المشاكل المعروفة في LLE هي مشكلة التثبيت. عندما يكون عدد الجيران أكبر من عدد أبعاد الإدخال، فإن المصفوفة التي تحدد كل حي محلي تكون ناقصة الرتبة. لمعالجة هذا الأمر، يطبق LLE القياسي معامل تثبيت عشوائي :math:`r`، والذي يتم اختياره بالنسبة لأثر مصفوفة الوزن المحلية. على الرغم من أنه يمكن إثباته رسميًا أنه عندما :math:`r \to 0`، فإن الحل يتقارب إلى التضمين المطلوب، لا يوجد ضمان بأنه سيتم العثور على الحل الأمثل لـ :math:`r > 0`. تظهر هذه المشكلة في عمليات التضمين التي تشوه الهندسة الأساسية للمنوع.

تتمثل إحدى طرق معالجة مشكلة التثبيت في استخدام متجهات وزن متعددة في كل حي. هذا هو جوهر *التضمين الخطي المحلي المعدل* (MLLE). يمكن إجراء MLLE باستخدام الدالة :func:`locally_linear_embedding` أو نظيرتها الموجهة للكائنات :class:`LocallyLinearEmbedding`، مع الكلمة الأساسية ``method = 'modified'``. يتطلب ``n_neighbors > n_components``.

.. figure:: ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_007.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

.. dropdown:: التعقيد

  يتكون خوارزم MLLE من ثلاث مراحل:

  1. **البحث عن الجيران الأقرب**. نفس LLE القياسي

  2. **بناء مصفوفة الوزن**. تقريبًا :math:`O[D N k^3] + O[N (k-D) k^2]`. المصطلح الأول مطابق تمامًا لذلك الخاص بـ LLE القياسي. يتعلق المصطلح الثاني ببناء مصفوفة الوزن من أوزان متعددة. في الواقع، تكون التكلفة الإضافية لبناء مصفوفة الوزن MLLE صغيرة نسبيًا مقارنة بتكلفة المرحلتين 1 و 3.

  3. **تحلل القيمة الذاتية الجزئي**. نفس LLE القياسي

  التعقيد العام لـ MLLE هو :math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[N (k-D) k^2] + O[d N^2]`.

  * :math:`N` : عدد نقاط بيانات التدريب
  * :math:`D` : البعد المدخلي
  * :math:`k` : عدد الجيران الأقرب
  * :math:`d` : البعد المخرجي

.. rubric:: المراجع

* `"MLLE: Modified Locally Linear Embedding Using Multiple Weights" <https://citeseerx.ist.psu.edu/doc_view/pid/0b060fdbd92cbcc66b383bcaa9ba5e5e624d7ee3>`_ Zhang, Z. & Wang, J.


تضمين هيشيان
============

تضمين هيشيان (المعروف أيضًا باسم LLE المستند إلى هيشيان: HLLE) هو طريقة أخرى لحل مشكلة التثبيت في LLE. إنها تدور حول شكل تربيعي قائم على هيشيان في كل حي يتم استخدامه لاستعادة البنية الخطية المحلية. على الرغم من أن عمليات التنفيذ الأخرى تلاحظ ضعفها في القياس مع حجم البيانات، فإن ``sklearn`` تنفذ بعض التحسينات الخوارزمية التي تجعل تكلفتها قابلة للمقارنة بتكلفة متغيرات LLE الأخرى للبعد المخرجي الصغير. يمكن إجراء HLLE باستخدام الدالة :func:`locally_linear_embedding` أو نظيرتها الموجهة للكائنات :class:`LocallyLinearEmbedding`، مع الكلمة الأساسية ``method = 'hessian'``. يتطلب ``n_neighbors > n_components * (n_components + 3) / 2``.

.. figure:: ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_008.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

.. dropdown:: التعقيد

يتكون خوارزم HLLE من ثلاث مراحل:

  1. **البحث عن الجيران الأقرب**. نفس LLE القياسي

  2. **بناء مصفوفة الوزن**. تقريبًا :math:`O[D N k^3] + O[N d^6]`. يعكس المصطلح الأول تكلفة مماثلة لتلك الخاصة بـ LLE القياسي. يأتي المصطلح الثاني من تحلل QR لمقدر هيشيان المحلي.

  3. **تحلل القيمة الذاتية الجزئي**. نفس LLE القياسي

  التعقيد العام لـ HLLE القياسي هو :math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[N d^6] + O[d N^2]`.

  * :math:`N` : عدد نقاط بيانات التدريب
  * :math:`D` : البعد المدخلي
  * :math:`k` : عدد الجيران الأقرب
  * :math:`d` : البعد المخرجي

.. rubric:: المراجع

* `"Hessian Eigenmaps: Locally linear embedding techniques for high-dimensional data" <http://www.pnas.org/content/100/10/5591>`_ Donoho, D. & Grimes, C. Proc Natl Acad Sci USA. 100:5591 (2003)

.. _spectral_embedding:

تضمين طيفي

التضمين الطيفي هو أسلوب لحساب التضمين غير الخطي. تقوم مكتبة Scikit-learn بتنفيذ خرائط لابلاس الخاصة، والتي تجد تمثيلاً منخفض الأبعاد للبيانات باستخدام تحليل طيفي لمخطط لابلاس. يمكن اعتبار المخطط الناتج كتقريب منفصل للمنوع منخفض الأبعاد في الفضاء عالي الأبعاد. يضمن تقليل دالة التكلفة المستندة إلى المخطط أن النقاط القريبة من بعضها البعض على المنوع يتم تعيينها بالقرب من بعضها البعض في الفضاء منخفض الأبعاد، مع الحفاظ على المسافات المحلية. يمكن إجراء التضمين الطيفي باستخدام الدالة :func:`spectral_embedding` أو نظيرتها الموجهة للكائنات :class:`SpectralEmbedding`.

.. dropdown:: التعقيد

  يتكون خوارزمية التضمين الطيفي (خرائط لابلاس الخاصة) من ثلاث مراحل:

  1. **بناء المخطط الموزون**. تحويل بيانات الإدخال الأولية إلى تمثيل المخطط باستخدام تمثيل مصفوفة التقارب (المجاورة).

  2. **بناء لابلاس للمخطط**. يتم بناء لابلاس للمخطط غير الطبيعي كـ :math:`L = D - A` وللابلاس الطبيعي كـ :math:`L = D^{-\frac{1}{2}} (D - A) D^{-\frac{1}{2}}`.

  3. **تحلل القيمة الذاتية الجزئي**. يتم إجراء تحلل القيمة الذاتية على لابلاس للمخطط.

  التعقيد العام للتضمين الطيفي هو :math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[d N^2]`.

  * :math:`N` : عدد نقاط بيانات التدريب
  * :math:`D` : البعد المدخلي
  * :math:`k` : عدد الجيران الأقرب
  * :math:`d` : البعد المخرجي

.. rubric:: المراجع

* `"خرائط لابلاس الخاصة لخفض الأبعاد وتمثيل البيانات" <https://web.cse.ohio-state.edu/~mbelkin/papers/LEM_NC_03.pdf>`_
  م. بلكين، ب. نيوغي، الحوسبة العصبية، يونيو 2003؛ 15 (6):1373-1396


محاذاة الفضاء التلامسي المحلي
=============================

على الرغم من أن محاذاة الفضاء التلامسي المحلي (LTSA) ليست تقنيًا متغيرًا من LLE، إلا أنها خوارزمية مشابهة بما يكفي لـ LLE بحيث يمكن وضعها في هذه الفئة. بدلاً من التركيز على الحفاظ على مسافات الجوار كما هو الحال في LLE، تسعى LTSA إلى وصف الهندسة المحلية في كل حي عبر فضائها التلامسي، وتقوم بتحسين عالمي لمحاذاة هذه الفضاءات التلامسية المحلية لتعلم التضمين. يمكن إجراء LTSA باستخدام الدالة :func:`locally_linear_embedding` أو نظيرتها الموجهة للكائنات :class:`LocallyLinearEmbedding`، مع الكلمة الرئيسية ``method = 'ltsa'``.

.. figure:: ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_009.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50

.. dropdown:: التعقيد

  تتكون خوارزمية LTSA من ثلاث مراحل:

  1. **بحث الجيران الأقرب**.  نفس LLE القياسي

  2. **بناء مصفوفة الوزن**. تقريبًا :math:`O[D N k^3] + O[k^2 d]`.  الحد الأول يعكس تكلفة مماثلة لتلك الخاصة بـ LLE القياسي.

  3. **تحلل القيمة الذاتية الجزئي**. نفس LLE القياسي

  التعقيد العام لـ LTSA القياسي هو :math:`O[D \log(k) N \log(N)] + O[D N k^3] + O[k^2 d] + O[d N^2]`.

  * :math:`N` : عدد نقاط بيانات التدريب
  * :math:`D` : البعد المدخلي
  * :math:`k` : عدد الجيران الأقرب
  * :math:`d` : البعد المخرجي

.. rubric:: المراجع

* :arxiv:`"المنوعات الرئيسية وخفض الأبعاد غير الخطي عبر محاذاة الفضاء التلامسي" <cs/0212008>`
  ز. زانغ، ه. زها، مجلة جامعة شنغهاي 8:406 (2004)

.. _multidimensional_scaling:

تحجيم متعدد الأبعاد (MDS)
===============================

`تدرج القياس متعدد الأبعاد <https://en.wikipedia.org/wiki/Multidimensional_scaling>`_
(:class:`MDS`) يسعى إلى تمثيل منخفض الأبعاد للبيانات بحيث تحترم المسافات جيدًا
المسافات في مساحة الأبعاد العالية الأصلية.

بشكل عام، :class:`MDS` هي تقنية تستخدم لتحليل بيانات التشابه أو الاختلاف. إنها تحاول نمذجة بيانات التشابه أو الاختلاف كمسافات في مساحات هندسية. يمكن أن تكون البيانات تصنيفات للتشابه بين الكائنات، أو ترددات تفاعل الجزيئات، أو مؤشرات التجارة بين البلدان.

هناك نوعان من خوارزمية MDS: متري وغير متري. في scikit-learn، تنفذ الفئة :class:`MDS` كلاهما. في MDS المتري، تنشأ مصفوفة التشابه المدخلة من مقياس (وبالتالي تحترم عدم المساواة الثلاثي)، ثم يتم تعيين المسافات بين نقطتي الإخراج لتكون قريبة قدر الإمكان من بيانات التشابه أو الاختلاف. في الإصدار غير المتري، ستحاول الخوارزميات الحفاظ على ترتيب المسافات، وبالتالي تبحث عن علاقة أحادية بين المسافات في المساحة المضمنة والتشابهات / الاختلافات.

.. figure:: ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_010.png
   :target: ../auto_examples/manifold/plot_lle_digits.html
   :align: center
   :scale: 50


دع :math:`S` تكون مصفوفة التشابه، و:math:`X` إحداثيات :math:`n` نقاط الإدخال. يتم تحويل أوجه التشابه :math:`\hat{d}_{ij}` بطرق مثالية معينة. ثم يتم تعريف الهدف، الذي يسمى الإجهاد، بواسطة :math:`\sum_{i < j} d_{ij}(X) - \hat{d}_{ij}(X)`


.. dropdown:: MDS المتري

  أبسط نموذج MDS متري، يسمى *MDS المطلق*، يتم تعريف أوجه عدم التطابق بواسطة
  :math:`\hat{d}_{ij} = S_{ij}`. مع MDS المطلق، يجب أن تتوافق القيمة :math:`S_{ij}`
  بالضبط مع المسافة بين النقطة :math:`i` و:math:`j` في نقطة التضمين.

  في أغلب الأحيان، يتم تعيين أوجه عدم التطابق على :math:`\hat{d}_{ij} = b S_{ij}`.

.. dropdown:: MDS غير المتري

  يركز MDS غير المتري على ترتيب البيانات. إذا
  :math:`S_{ij} > S_{jk}`, إذن يجب أن يفرض التضمين :math:`d_{ij} <
 d_{jk}`. لهذا السبب، نناقشها من حيث الاختلافات
  (:math:`\delta_{ij}`) بدلاً من أوجه التشابه (:math:`S_{ij}`). لاحظ أنه
  يمكن الحصول على الاختلافات بسهولة من أوجه التشابه من خلال تحويل بسيط، على سبيل المثال. :math:`\delta_{ij}=c_1-c_2 S_{ij}` لبعض الثوابت الحقيقية
  :math:`c_1, c_2`. خوارزمية بسيطة لفرض الترتيب الصحيح هي استخدام
  انحدار أحادي من :math:`d_{ij}` على :math:`\delta_{ij}`, مما ينتج عنه أوجه عدم التطابق :math:`\hat{d}_{ij}` بنفس ترتيب :math:`\delta_{ij}`.

  الحل التافه لهذه المشكلة هو تعيين جميع النقاط على الأصل. من أجل
  تجنب ذلك، يتم تطبيع أوجه عدم التطابق :math:`\hat{d}_{ij}`. ملاحظة
  نظرًا لأننا نهتم فقط بالترتيب النسبي، يجب أن يكون هدفنا
  غير حساس للترجمة البسيطة والتحجيم، ومع ذلك فإن الإجهاد المستخدم في MDS المتري
  حساس للتحجيم. لمعالجة هذا الأمر، قد يستخدم MDS غير المتري
  إجهادًا عاديًا، يُعرف باسم Stress-1 المحدد على أنه

  .. math::
      \sqrt{\frac{\sum_{i < j} (d_{ij} - \hat{d}_{ij})^2}{\sum_{i < j} d_{ij}^2}}.

  يمكن تمكين استخدام Stress-1 العادي عن طريق تعيين `normalized_stress=True`,
  ومع ذلك، فهو متوافق فقط مع مشكلة MDS غير المتري وسيتم تجاهله
  في الحالة المتري.

  .. figure:: ../auto_examples/manifold/images/sphx_glr_plot_mds_001.png
    :target: ../auto_examples/manifold/plot_mds.html
    :align: center
    :scale: 60

.. rubric:: المراجع

* `"تدرج القياس متعدد الأبعاد الحديث - النظرية والتطبيقات"
  <https://www.springer.com/fr/book/9780387251509>`_
  بورغ، I.؛ غروينين P. سلسلة سبرينغر في الإحصاء (1997)

* `"تدرج القياس متعدد الأبعاد غير المتري: طريقة رقمية"
  <http://cda.psych.uiuc.edu/psychometrika_highly_cited_articles/kruskal_1964b.pdf>`_
  كروسكال، J. سيكومتريكا، 29 (1964)

* `"تدرج القياس متعدد الأبعاد عن طريق تحسين جود التوافق مع فرضية غير متري"
  <http://cda.psych.uiuc.edu/psychometrika_highly_cited_articles/kruskal_1964a.pdf>`_
  كروسكال، J. سيكومتريكا، 29، (1964)

.. _t_sne:

تدرج القياس متعدد الأبعاد الموزع t (t-SNE)
    

  تقوم t-SNE (TSNE) بتحويل تقاربات نقاط البيانات إلى احتمالات. يتم تمثيل التقاربات في الفضاء الأصلي باحتمالات مشتركة غاوسية، بينما يتم تمثيل التقاربات في الفضاء المضمن باحتمالات توزيع t للطالب. يسمح هذا لـ t-SNE بأن تكون حساسة بشكل خاص للهيكل المحلي ولديها بعض المزايا الأخرى على التقنيات الحالية:

    * الكشف عن الهيكل على العديد من المقاييس على خريطة واحدة
    * الكشف عن البيانات التي تقع في العديد من الحيزات أو العناقيد المختلفة
    * تقليل الميل إلى تجميع النقاط معًا في المركز

    في حين أن Isomap و LLE والمتغيرات هي الأنسب لنشر حيز مستمر منخفض الأبعاد واحد، سيركز t-SNE على الهيكل المحلي للبيانات وسيكون ميالًا لاستخراج مجموعات محلية مجمعة من العينات كما هو موضح في مثال منحنى S. قد تكون هذه القدرة على تجميع العينات بناءً على الهيكل المحلي مفيدة لفك تشابك مجموعة بيانات بصريًا تتضمن العديد من الحيزات في وقت واحد كما هو الحال في مجموعة بيانات الأرقام.

    سيتم تقليل تباعد Kullback-Leibler (KL) للاحتمالات المشتركة في الفضاء الأصلي والفضاء المضمن عن طريق الانحدار التدرجي. لاحظ أن تباعد KL ليس محدبًا، أي أن عمليات إعادة التشغيل المتعددة مع عمليات تهيئة مختلفة ستنتهي في الحد الأدنى المحلي لتباعد KL. وبالتالي، من المفيد أحيانًا تجربة بذور مختلفة واختيار التضمين بأقل تباعد KL.

    عيوب استخدام t-SNE هي تقريبًا:

    * t-SNE مكلف حسابيًا، ويمكن أن يستغرق عدة ساعات في مجموعات البيانات التي تضم ملايين العينات حيث ستنتهي PCA في ثوانٍ أو دقائق
    * تقتصر طريقة Barnes-Hut t-SNE على عمليات التضمين ثنائية أو ثلاثية الأبعاد.
    * الخوارزمية عشوائية ويمكن أن تؤدي عمليات إعادة التشغيل المتعددة ببذور مختلفة إلى عمليات تضمين مختلفة. ومع ذلك، من المشروع تمامًا اختيار التضمين بأقل خطأ.
    * لا يتم الحفاظ على البنية العالمية بشكل صريح. يتم التخفيف من هذه المشكلة عن طريق تهيئة النقاط باستخدام PCA (باستخدام `init='pca'`).

    .. figure:: ../auto_examples/manifold/images/sphx_glr_plot_lle_digits_013.png
       :target: ../auto_examples/manifold/plot_lle_digits.html
       :align: center
       :scale: 50

    .. dropdown:: تحسين t-SNE

      الغرض الرئيسي من t-SNE هو تصور البيانات عالية الأبعاد. وبالتالي، فإنه يعمل بشكل أفضل عندما يتم تضمين البيانات في بعدين أو ثلاثة أبعاد.

      قد يكون تحسين تباعد KL صعبًا بعض الشيء في بعض الأحيان. هناك خمسة معلمات تتحكم في تحسين t-SNE وبالتالي ربما جودة التضمين الناتج:

      * الحيرة
      * عامل المبالغة المبكر
      * معدل التعلم
      * الحد الأقصى لعدد التكرارات
      * الزاوية (غير مستخدمة في الطريقة الدقيقة)

      يتم تعريف الحيرة على أنها :math:`k=2^{(S)}` حيث :math:`S` هو إنتروبيا شانون لتوزيع الاحتمالات الشرطي. حيرة :math:`k`-sided die هي :math:`k`، بحيث يكون :math:`k` فعليًا عدد الجيران الأقرب الذين يفكر فيهم t-SNE عند إنشاء الاحتمالات الشرطية. تؤدي الحيرة الأكبر إلى المزيد من الجيران الأقرب وأقل حساسية للهيكل الصغير. على العكس من ذلك، فإن الحيرة الأقل تأخذ في الاعتبار عددًا أقل من الجيران، وبالتالي تتجاهل المزيد من المعلومات العالمية لصالح الحي المحلي. مع زيادة أحجام مجموعة البيانات، ستكون هناك حاجة إلى المزيد من النقاط للحصول على عينة معقولة من الحي المحلي، وبالتالي قد تكون هناك حاجة إلى حيرة أكبر. وبالمثل، ستتطلب مجموعات البيانات الأكثر ضوضاءً قيم حيرة أكبر لتضمين ما يكفي من الجيران المحليين لرؤية ما وراء الضوضاء الخلفية.

      عادةً ما يكون الحد الأقصى لعدد التكرارات مرتفعًا بدرجة كافية ولا يحتاج إلى أي ضبط. يتكون التحسين من مرحلتين: مرحلة المبالغة المبكرة والتحسين النهائي. خلال مرحلة المبالغة المبكرة، سيتم زيادة الاحتمالات المشتركة في الفضاء الأصلي بشكل مصطنع عن طريق الضرب بعامل معين. تؤدي العوامل الأكبر إلى فجوات أكبر بين العناقيد الطبيعية في البيانات. إذا كان العامل مرتفعًا جدًا، فقد يزيد تباعد KL خلال هذه المرحلة. عادةً لا يتعين ضبطه. المعلمة الحرجة هي معدل التعلم. إذا كان منخفضًا جدًا، فسيتم تعليق الانحدار التدرجي في حد أدنى محلي سيئ. إذا كان مرتفعًا جدًا، سيزداد تباعد KL أثناء التحسين. الإرشاد المقترح في Belkina et al. (2019) هو تعيين معدل التعلم لحجم العينة مقسومًا على عامل المبالغة المبكر. ننفذ هذا الإرشاد كحجة `learning_rate='auto'`. يمكن العثور على المزيد من النصائح في الأسئلة الشائعة لـ Laurens van der Maaten (انظر المراجع). المعلمة الأخيرة، الزاوية، هي مقايضة بين الأداء والدقة. تعني الزوايا الأكبر أنه يمكننا تقريب مناطق أكبر بنقطة واحدة، مما يؤدي إلى سرعة أفضل ولكن نتائج أقل دقة.

      يوفر `"How to Use t-SNE Effectively" <https://distill.pub/2016/misread-tsne/>`_ مناقشة جيدة لآثار المعلمات المختلفة، بالإضافة إلى مخططات تفاعلية لاستكشاف آثار المعلمات المختلفة.

    .. dropdown:: Barnes-Hut t-SNE
    
التنفيذ الخاص بـ Barnes-Hut t-SNE الذي تم تنفيذه هنا عادة ما يكون أبطأ بكثير من خوارزميات تعلم المنوعات الأخرى. التحسين صعب للغاية وحساب التدرج هو :math:`O[d N log(N)]`، حيث :math:`d` هو عدد أبعاد الإخراج و :math:`N` هو عدد العينات. تحسن طريقة Barnes-Hut على الطريقة الدقيقة حيث تعقيد t-SNE هو :math:`O[d N^2]`, ولكن لديها العديد من الاختلافات الملحوظة الأخرى:

* يعمل تنفيذ Barnes-Hut فقط عندما يكون البعد المستهدف 3 أو أقل. الحالة ثنائية الأبعاد نموذجية عند بناء التصورات.
* يعمل Barnes-Hut فقط مع بيانات الإدخال الكثيفة. يمكن تضمين مصفوفات البيانات المتفرقة فقط بالطريقة الدقيقة أو يمكن تقريبها بإسقاط رتبة منخفض كثيف على سبيل المثال باستخدام :class:`~sklearn.decomposition.PCA`
* Barnes-Hut هو تقريب للطريقة الدقيقة. يتم معلمتة التقريب باستخدام معلمة الزاوية، وبالتالي فإن معلمة الزاوية غير مستخدمة عندما method="exact"
* Barnes-Hut قابل للتطوير بشكل كبير. يمكن استخدام Barnes-Hut لتضمين مئات الآلاف من نقاط البيانات بينما يمكن للطريقة الدقيقة التعامل مع آلاف العينات قبل أن تصبح حسابياً غير قابلة للحل

لأغراض التصور (وهو حالة الاستخدام الرئيسية لـ t-SNE)، يوصى بشدة باستخدام طريقة Barnes-Hut. طريقة t-SNE الدقيقة مفيدة للتحقق من الخصائص النظرية للتضمين ربما في مساحة أعلى الأبعاد ولكن تقتصر على مجموعات البيانات الصغيرة بسبب القيود الحسابية.

لاحظ أيضًا أن تسميات الأرقام تتطابق تقريبًا مع التجميع الطبيعي الذي تم العثور عليه بواسطة t-SNE بينما ينتج الإسقاط الخطي ثنائي الأبعاد لنموذج PCA تمثيلًا تتداخل فيه مناطق التسمية إلى حد كبير. هذا دليل قوي على أنه يمكن فصل هذه البيانات جيدًا بواسطة طرق غير خطية تركز على البنية المحلية (على سبيل المثال، SVM مع نواة RBF غاوسية). ومع ذلك، فإن الفشل في تصور مجموعات متجانسة ذات تسميات جيدة الانفصال مع t-SNE في البعدين لا يعني بالضرورة أن البيانات لا يمكن تصنيفها بشكل صحيح بواسطة نموذج خاضع للإشراف. قد تكون الحالة أن البعدين ليسا مرتفعين بما يكفي لتمثيل البنية الداخلية للبيانات بدقة.

.. rubric:: المراجع

* `"Visualizing High-Dimensional Data Using t-SNE"
  <https://jmlr.org/papers/v9/vandermaaten08a.html>`_
  van der Maaten, L.J.P.; Hinton, G. Journal of Machine Learning Research (2008)

* `"t-Distributed Stochastic Neighbor Embedding"
  <https://lvdmaaten.github.io/tsne/>`_ van der Maaten, L.J.P.

* `"Accelerating t-SNE using Tree-Based Algorithms"
  <https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf>`_
  van der Maaten, L.J.P.; Journal of Machine Learning Research 15(Oct):3221-3245, 2014.

* `"Automated optimized parameters for T-distributed stochastic neighbor
  embedding improve visualization and analysis of large datasets"
  <https://www.nature.com/articles/s41467-019-13055-y>`_
  Belkina, A.C., Ciccolella, C.O., Anno, R., Halpert, R., Spidlen, J.,
  Snyder-Cappione, J.E., Nature Communications 10, 5415 (2019).

نصائح حول الاستخدام العملي
=====================

* تأكد من استخدام نفس المقياس على جميع الميزات. نظرًا لأن طرق تعلم المنوعات تعتمد على بحث الجار الأقرب ، فقد يؤدي ذلك إلى ضعف أداء الخوارزمية. انظر :ref:`StandardScaler <preprocessing_scaler>` للحصول على طرق ملائمة لتغيير حجم البيانات غير المتجانسة.

* يمكن استخدام خطأ إعادة البناء الذي تحسبه كل روتين لاختيار البعد الأمثل للإخراج. بالنسبة لمنوعات ذات أبعاد :math:`d` مضمنة في مساحة معلمات ذات أبعاد :math:`D` ، سينخفض خطأ إعادة البناء مع زيادة ``n_components`` حتى ``n_components == d``.

* لاحظ أن البيانات الضوضائية يمكن أن "تقصر" المنوعات ، بشكل أساسي تعمل كجسر بين أجزاء من المنوعات التي ستكون منفصلة جيدًا. تعلم المنوعات على البيانات الضوضائية و / أو غير المكتملة هو مجال بحث نشط.

* يمكن أن تؤدي تكوينات الإدخال معينة إلى مصفوفات وزن مفردة ، على سبيل المثال عندما تكون أكثر من نقطتين في مجموعة البيانات متطابقة ، أو عندما يتم تقسيم البيانات إلى مجموعات منفصلة. في هذه الحالة ، سيفشل ``solver='arpack'`` في العثور على الفضاء الخالي. أسهل طريقة لمعالجة هذا الأمر هي استخدام ``solver='dense'`` الذي سيعمل على مصفوفة مفردة ، على الرغم من أنه قد يكون بطيئًا جدًا اعتمادًا على عدد نقاط الإدخال. بدلاً من ذلك ، يمكن للمرء أن يحاول فهم مصدر التفرد: إذا كان ذلك بسبب مجموعات منفصلة ، فإن زيادة ``n_neighbors`` قد تساعد. إذا كان ذلك بسبب نقاط متطابقة في مجموعة البيانات ، فإن إزالة هذه النقاط قد يساعد.

.. seealso::

   :ref:`random_trees_embedding` يمكن أن يكون مفيدًا أيضًا لاستخلاص تمثيلات غير خطية لمساحة الميزة ، كما أنه لا يؤدي إلى تقليل الأبعاد.

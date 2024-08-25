
.. _cross_decomposition:

===================
تحلل تقاطعي

===================

.. currentmodule:: sklearn.cross_decomposition

يحتوي نموذج التحلل المتقاطع على تقديرات خاضعة للإشراف من أجل تقليل الأبعاد والانحدار، والتي تنتمي إلى عائلة "الحد الأدنى الجزئي من المربعات".

.. figure:: ../auto_examples/cross_decomposition/images/sphx_glr_plot_compare_cross_decomposition_001.png
   :target: ../auto_examples/cross_decomposition/plot_compare_cross_decomposition.html
   :scale: 75%
   :align: center


تبحث خوارزميات التحلل المتقاطع عن العلاقات الأساسية بين مصفوفتين (X و Y). وهي نهج متغير كامن لنمذجة هياكل التباين في هاتين المساحتين. ستحاول العثور على الاتجاه متعدد الأبعاد في مساحة X الذي يفسر أقصى اتجاه للتباين متعدد الأبعاد في مساحة Y. بمعنى آخر، يرسم PLS كل من `X` و `Y` في فضاء ذي أبعاد أقل بحيث يكون التباين بين `transformed(X)` و `transformed(Y)` هو الأقصى.

يظهر PLS أوجه تشابه مع "انحدار المكونات الرئيسية" (PCR)، حيث يتم أولاً عرض العينات في فضاء ذي أبعاد أقل، ويتم التنبؤ بالأهداف `y` باستخدام `transformed(X)`. إحدى المشكلات في PCR هي أن تقليل الأبعاد غير خاضع للإشراف، وقد تفقد بعض المتغيرات المهمة: سيحتفظ PCR بالميزات التي لديها أكبر تباين، ولكن من الممكن أن تكون الميزات ذات التباينات الصغيرة ذات صلة بالتنبؤ بالهدف. بطريقة ما، يسمح PLS بنفس نوع تقليل الأبعاد، ولكن مع مراعاة الأهداف `y`. يتم تقديم توضيح لهذه الحقيقة في المثال التالي:
* :ref:`sphx_glr_auto_examples_cross_decomposition_plot_pcr_vs_pls.py`.

بصرف النظر عن CCA، فإن تقديرات PLS مناسبة بشكل خاص عندما تحتوي مصفوفة المتنبئات على متغيرات أكثر من المشاهدات، وعندما يكون هناك تعدد في الارتباط بين الميزات. في المقابل، سيفشل الانحدار الخطي القياسي في هذه الحالات ما لم يتم تنظيمه.

الطبقات المضمنة في هذا النموذج هي: :class:`PLSRegression`، :class:`PLSCanonical`، :class:`CCA` و :class:`PLSSVD`

PLSCanonical
------------

سنصف هنا الخوارزمية المستخدمة في :class:`PLSCanonical`. تستخدم التقديرات الأخرى المتغيرات من هذه الخوارزمية، ويتم تفصيلها أدناه. نوصي القسم [1]_ للحصول على مزيد من التفاصيل والمقارنات بين هذه الخوارزميات. في [1]_، :class:`PLSCanonical` يتوافق مع "PLSW2A".

بالنظر إلى مصفوفتين مركّزتين :math:`X \in \mathbb{R}^{n \times d}` و :math:`Y \in \mathbb{R}^{n \times t}`, وعدد المكونات :math:`K`، :class:`PLSCanonical` يعمل كالتالي:

تعيين :math:`X_1` إلى :math:`X` و :math:`Y_1` إلى :math:`Y`. ثم، لكل :math:`k \in [1، K]`:

- أ) حساب :math:`u_k \in \mathbb{R}^d` و :math:`v_k \in \mathbb{R}^t`، متجهات التفرد الأيسر والأيمن الأولى لمصفوفة التغاير المتبادل :math:`C = X_k^T Y_k`. :math:`u_k` و :math:`v_k` يطلق عليهما *الأوزان*. بحكم التعريف، يتم اختيار :math:`u_k` و :math:`v_k` بحيث يزيدان من التباين بين :math:`X_k` المتوقعة والهدف المتوقع، أي :math:`\text{Cov}(X_k u_k، Y_k v_k)`.
- ب) إسقاط :math:`X_k` و :math:`Y_k` على متجهات التفرد للحصول على *الدرجات*: :math:`\xi_k = X_k u_k` و :math:`\omega_k = Y_k v_k`
- ج) إرجاع :math:`X_k` على :math:`\xi_k`، أي إيجاد متجه :math:`\gamma_k \in \mathbb{R}^d` بحيث تكون المصفوفة من الرتبة 1 :math:`\xi_k \gamma_k^T` قريبة قدر الإمكان من :math:`X_k`. افعل نفس الشيء على :math:`Y_k` مع :math:`\omega_k` للحصول على :math:`\delta_k`. المتجهات :math:`\gamma_k` و :math:`\delta_k` تسمى *التحميلات*.
- د) *نفخ* :math:`X_k` و :math:`Y_k`، أي طرح التقريب من الرتبة 1: :math:`X_{k+1} = X_k - \xi_k \gamma_k^T`، و :math:`Y_{k + 1} = Y_k - \omega_k \delta_k^T`.

في النهاية، قمنا بتقريب :math:`X` كمجموع المصفوفات من الرتبة 1: :math:`X = \Xi \Gamma^T` حيث :math:`\Xi \in \mathbb{R}^{n \times K}` يحتوي على الدرجات في أعمدته، و :math:`\Gamma^T \in \mathbb{R}^{K \times d}` يحتوي على التحميلات في صفوفه. وبالمثل بالنسبة لـ :math:`Y`، لدينا :math:`Y = \Omega \Delta^T`.

لاحظ أن مصفوفات الدرجات :math:`\Xi` و :math:`\Omega` تتوافق مع إسقاطات بيانات التدريب :math:`X` و :math:`Y`، على التوالي.

قد يتم تنفيذ الخطوة * أ) * بطريقتين: إما عن طريق حساب SVD الكامل لـ :math:`C` والاحتفاظ فقط بمتجهات التفرد ذات أكبر القيم التفردية، أو عن طريق حساب متجهات التفرد مباشرةً باستخدام طريقة القوة (راجع القسم 11.3 في [1]_)، والتي تتوافق مع خيار `'nipals'` من معلمة `algorithm`.

.. dropdown:: تحويل البيانات

  لتحويل :math:`X` إلى :math:`\bar{X}`، نحتاج إلى إيجاد مصفوفة الإسقاط :math:`P` بحيث :math:`\bar{X} = XP`. نعلم أنه بالنسبة لبيانات التدريب، :math:`\Xi = XP`، و :math:`X = \Xi \Gamma^T`. بتعيين :math:`P = U(\Gamma^T U)^{-1}` حيث :math:`U` هي المصفوفة ذات :math:`u_k` في الأعمدة، لدينا :math:`XP = X U(\Gamma^T U)^{-1} = \Xi (\Gamma^T U) (\Gamma^T U)^{-1} = \Xi` كما هو مطلوب. يمكن الوصول إلى مصفوفة الدوران :math:`P` من سمة `x_rotations_`.

  وبالمثل، يمكن تحويل :math:`Y` باستخدام مصفوفة الدوران :math:`V(\Delta^T V)^{-1}`, التي يمكن الوصول إليها عبر سمة `y_rotations_`.

.. dropdown:: التنبؤ بالأهداف `Y`

  للتنبؤ بأهداف بعض البيانات :math:`X`، نحن نبحث عن مصفوفة المعامل :math:`\beta \in R^{d \times t}` بحيث :math:`Y = X\beta`.

  الفكرة هي محاولة التنبؤ بالأهداف المحولة :math:`\Omega` كدالة للعينات المحولة :math:`\Xi`، عن طريق حساب :math:`\alpha \in \mathbb{R}` بحيث :math:`\Omega = \alpha \Xi`.

  ثم، لدينا :math:`Y = \Omega \Delta^T = \alpha \Xi \Delta^T`، ونظرًا لأن :math:`\Xi` هي بيانات التدريب المحولة، لدينا أن :math:`Y = X \alpha P \Delta^T`، ونتيجة لذلك فإن مصفوفة المعامل :math:`\beta = \alpha P \Delta^T`.

  يمكن الوصول إلى :math:`\beta` من خلال سمة `coef_`.


----

    :class:`PLSSVD` هو نسخة مبسطة من :class:`PLSCanonical` الذي تم وصفه سابقًا: بدلاً من تقليل مصفوفات :math:`X_k` و :math:`Y_k` تكراريًا، فإن :class:`PLSSVD` يحسب SVD لـ :math:`C = X^TY` مرة واحدة فقط، ويخزن متجهات التفرد `n_components` المقابلة لأكبر القيم التفردية في المصفوفتين `U` و `V`، والتي تقابل الخاصيتين `x_weights_` و `y_weights_`. هنا، يتم تحويل البيانات ببساطة إلى `transformed(X) = XU` و `transformed(Y) = YV`.

    إذا كان `n_components == 1`، فإن :class:`PLSSVD` و :class:`PLSCanonical` يكونان مكافئين تمامًا.

    PLSRegression
    -------------

    إن المقدّر :class:`PLSRegression` مشابه لـ :class:`PLSCanonical` مع `algorithm='nipals'`، مع وجود اختلافين رئيسيين:

    - في الخطوة أ) في طريقة القوة لحساب :math:`u_k` و :math:`v_k`، لا يتم تطبيع :math:`v_k` أبدًا.
    - في الخطوة ج)، يتم تقريب الأهداف :math:`Y_k` باستخدام إسقاط :math:`X_k` (أي :math:`\xi_k`) بدلاً من إسقاط :math:`Y_k` (أي :math:`\omega_k`). بمعنى آخر، حساب الحمولة مختلف. نتيجة لذلك، سيتأثر أيضًا الانكماش في الخطوة د).

    هاتان التعديلان يؤثران على خرج `predict` و `transform`، والتي ليست هي نفسها بالنسبة لـ :class:`PLSCanonical`. أيضًا، بينما يقتصر عدد المكونات على `min(n_samples, n_features, n_targets)` في :class:`PLSCanonical`، هنا تكون الحدود هي رتبة :math:`X^TX`، أي `min(n_samples, n_features)`.

    :class:`PLSRegression` معروف أيضًا باسم PLS1 (أهداف فردية) و PLS2 (أهداف متعددة). مثل :class:`~sklearn.linear_model.Lasso`، فإن :class:`PLSRegression` هو شكل من أشكال الانحدار الخطي المنتظم حيث يتحكم عدد المكونات في قوة التنظيم.

    تحليل الارتباطات القياسية
    ------------------------------

    تم تطوير تحليل الارتباطات القياسية قبل PLS وبشكل مستقل عنها. لكن اتضح أن :class:`CCA` هي حالة خاصة من PLS، وتوافق PLS في "الوضع B" في الأدبيات.

    يختلف :class:`CCA` عن :class:`PLSCanonical` في طريقة حساب الأوزان :math:`u_k` و :math:`v_k` في طريقة القوة للخطوة أ). يمكن العثور على التفاصيل في القسم 10 من [1]_.

    نظرًا لأن :class:`CCA` تنطوي على انقلاب :math:`X_k^TX_k` و :math:`Y_k^TY_k`، يمكن أن يكون هذا المقدّر غير مستقر إذا كان عدد الميزات أو الأهداف أكبر من عدد العينات.

    .. rubric:: المراجع

    .. [1] `A survey of Partial Least Squares (PLS) methods, with emphasis on the two-block case <https://stat.uw.edu/sites/default/files/files/reports/2000/tr371.pdf>`_، JA Wegelin

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_cross_decomposition_plot_compare_cross_decomposition.py`
    * :ref:`sphx_glr_auto_examples_cross_decomposition_plot_pcr_vs_pls.py`
    
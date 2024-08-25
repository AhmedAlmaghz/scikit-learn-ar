  هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرياضية والروابط والتاجات والشفرة البرمجية دون ترجمة:

.. _data-transforms:

تحويلات مجموعة البيانات
-----------------------

يوفر سكيت-ليرن مكتبة من المحولات، والتي يمكنها تنظيف (انظر :ref:`preprocessing`)، وتقليل (انظر :ref:`data_reduction`)، وتوسيع (انظر :ref:`kernel_approximation`)، أو توليد (انظر :ref:`feature_extraction`) تمثيلات للميزات.

مثل غيرها من المقدرين، يتم تمثيل هذه المحولات بفئات مع طريقة "fit"، والتي تتعلم معلمات النموذج (مثل المتوسط والانحراف المعياري للتطبيع) من مجموعة التدريب، وطريقة "transform" التي تطبق نموذج التحويل هذا على بيانات غير مرئية. قد يكون "fit_transform" أكثر ملاءمة وكفاءة لنمذجة وتحويل بيانات التدريب في وقت واحد.

يتم تغطية الجمع بين هذه المحولات، سواء بالتوازي أو التسلسل، في :ref:`combining_estimators`. يغطي :ref:`metrics` تحويل مساحات الميزات إلى مصفوفات تقارب، بينما يعتبر :ref:`preprocessing_targets` تحويلات مساحة الهدف (مثل التسميات الفئوية) للاستخدام في سكيت-ليرن.

.. toctree::
    :maxdepth: 2

    modules/compose
    modules/feature_extraction
    modules/preprocessing
    modules/impute
    modules/unsupervised_reduction
    modules/random_projection
    modules/kernel_approximation
    modules/metrics
    modules/preprocessing_targets

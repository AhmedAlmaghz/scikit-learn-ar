هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية:

.. _toy_datasets:

مجموعات بيانات تجريبية
====================

.. currentmodule:: sklearn.datasets

تأتي مكتبة scikit-learn مع بعض مجموعات البيانات القياسية الصغيرة التي لا تتطلب تنزيل أي ملف من أي موقع ويب خارجي.

يمكن تحميلها باستخدام الدوال التالية:

.. autosummary::

   load_iris
   load_diabetes
   load_digits
   load_linnerud
   load_wine
   load_breast_cancer

هذه المجموعات مفيدة للتوضيح السريع لسلوك الخوارزميات المختلفة المنفذة في scikit-learn. ومع ذلك، فهي غالبًا ما تكون صغيرة جدًا بحيث لا تكون ممثلة لمهام التعلم الآلي في العالم الحقيقي.

.. include:: ../../sklearn/datasets/descr/iris.rst

.. include:: ../../sklearn/datasets/descr/diabetes.rst

.. include:: ../../sklearn/datasets/descr/digits.rst

.. include:: ../../sklearn/datasets/descr/linnerud.rst

.. include:: ../../sklearn/datasets/descr/wine_data.rst

.. include:: ../../sklearn/datasets/descr/breast_cancer.rst

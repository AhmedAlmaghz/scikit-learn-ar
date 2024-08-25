===================
انحدار متساوي التوتر
===================

.. currentmodule:: sklearn.isotonic

تقوم فئة :class:`IsotonicRegression` بتناسب دالة حقيقية غير متناقصة مع بيانات أحادية البعد. إنها تحل المشكلة التالية:

.. math::
    \min \sum_i w_i (y_i - \hat{y}_i)^2

مع مراعاة :math:`\hat{y}_i \le \hat{y}_j` كلما :math:`X_i \le X_j`,
حيث الأوزان :math:`w_i` موجبة بشكل صارم، وكلا من `X` و `y` كميات حقيقية تعسفية.

يغير معامل `increasing` القيد إلى
:math:`\hat{y}_i \ge \hat{y}_j` كلما :math:`X_i \le X_j`. تعيينه إلى
'auto' سيختار القيد تلقائيًا بناءً على `معامل ارتباط الرتب لسبيرمان
<https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_.

تنتج :class:`IsotonicRegression` سلسلة من التنبؤات
:math:`\hat{y}_i` لبيانات التدريب التي هي الأقرب إلى الأهداف
:math:`y` من حيث متوسط الخطأ المربع. يتم استقراء هذه التنبؤات
لتوقع البيانات غير المرئية. وبالتالي فإن تنبؤات :class:`IsotonicRegression`
تشكل دالة خطية على مراحل:

.. figure:: ../auto_examples/miscellaneous/images/sphx_glr_plot_isotonic_regression_001.png
   :target: ../auto_examples/miscellaneous/plot_isotonic_regression.html
   :align: center

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_miscellaneous_plot_isotonic_regression.py`
    

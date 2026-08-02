# src/

The real, reusable package. Everything here is either (a) promoted, generalized code
that started life in learning/, or (b) new code that only makes sense at real scale
(training loops, evaluation, the dataset wrapper). Nothing in src/ should duplicate
logic that lives only in learning/ — once something is promoted, the src/ version is
the one used everywhere downstream (notebooks/, results/ generation, etc.).

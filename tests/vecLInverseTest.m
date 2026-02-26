function tests = vecLInverseTest
addpath("../src")
tests = functiontests(localfunctions);
end

% Check answer for a simple example
function testCompareKnown(testCase)
    v = (1:6)';
    calculated = vecLInverse(v);
    expected = [[1 1 2 3];
                [1 1 4 5];
                [2 4 1 6];
                [3 5 6 1]];
    verifyEqual(testCase,calculated,expected);
end


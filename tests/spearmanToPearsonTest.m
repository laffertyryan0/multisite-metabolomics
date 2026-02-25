function tests = spearmanToPearsonTest
addpath("../src")
tests = functiontests(localfunctions);
end

% Check that on average, spearmanToPearson(spear) is at least closer
% to pears than spear itself
function testConversionCorrect(testCase)
    diff = 0;
    better = 0;
    for i=1:1000
        x = randn(500,1);
        y = 4*x + 3*randn(500,1);
        pears = corr(x,y,'Type','Pearson');
        spear = corr(x,y,'Type','Spearman');
        diff = diff + spear-pears;
        better = better + spearmanToPearson(spear)-pears;
    end
    diff = diff/1000;
    better = better/1000;
    verifyLessThan(testCase,abs(better),abs(diff));
end
//C:\Users\Brian\IdeaProjects\WormsAI\store\extensions

package net.chrono7.wormsai;

import net.lightbody.bmp.BrowserMobProxy;
import net.lightbody.bmp.BrowserMobProxyServer;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.openqa.selenium.*;
import org.openqa.selenium.Dimension;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.remote.CapabilityType;

import java.awt.*;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.InputEvent;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.apache.commons.io.FileUtils.writeStringToFile;

public class WebDriverExecutor {

    private static final int PIXELS_RIGHT = 5;
    private static final int PIXELS_LEFT = 35;
    private static final int PIXELS_DOWN = 80;
    private static final int PIXELS_UP = 110;
    private static final Dimension WINDOW_SIZE = new Dimension(1920, 1080);
    private Rectangle gameRect;
    private Rectangle boundsRect;
    private ChromeDriver driver;
    private WebElement game;
    private Robot robot;
    private int lastMouseX, lastMouseY;
    private FFmpegFrameGrabber grabber;

    WebDriverExecutor() {

        BrowserMobProxy proxy = new BrowserMobProxyServer();
        proxy.start(18904);
        proxy.blacklistRequests("http://slither.io/s/bg54.jpg", 204);
        proxy.blacklistRequests("http://slither.io/s/gbg.jpg", 204);
        //Finish setting up your driver


        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            proxy.stop();
            driver.close();
            System.exit(0);
        }));

        System.setProperty("webdriver.chrome.driver", "C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\drivers\\chromedriver.exe");

        // Get BrowserMobProxy server port for PAC file
        int proxyPort = proxy.getPort();

        // Create PAC file to send WebSocket requests direct but other protocols throught BrowserMobProxy server
        String pacFunction = "function FindProxyForURL(url, host) { if (url.substring(0, 3) === \"ws:\" || url.substring(0, 4) === \"wss:\") { " +
                "return \"DIRECT\"; } else { return \"PROXY 127.0.0.1:" + proxyPort + "\"; } }";
        File pacFile = new File("C:/Users/Brian/IdeaProjects/WormsAI/store/misc/proxy.pac");
        try {
            writeStringToFile(pacFile, pacFunction, "UTF-8");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Set PAC file path to Selenium proxy
        Proxy seleniumProxy = new Proxy();
        seleniumProxy.setProxyType(Proxy.ProxyType.PAC);
        seleniumProxy.setProxyAutoconfigUrl(String.valueOf(pacFile.toURI()));

        ChromeOptions options = new ChromeOptions();
        options.setExperimentalOption("useAutomationExtension", false);
        options.setExperimentalOption("excludeSwitches",
                Collections.singletonList("enable-automation"));
        options.setCapability(CapabilityType.PROXY, seleniumProxy);

        driver = new ChromeDriver(options);

        try {
            robot = new Robot();
        } catch (AWTException e) {
            e.printStackTrace();
        }
    }

    private Point boundMousePoint(int inX, int inY) {
        int x = Math.max(Math.min(boundsRect.width, inX), boundsRect.x);
        int y = Math.max(Math.min(boundsRect.height, inY), boundsRect.y);

        return new Point(x, y);
    }

    public void act(GameInstruction action) {
        if (Directions.RELATIVE) {
            Point curPoint = MouseInfo.getPointerInfo().getLocation();
            point(boundMousePoint(curPoint.x + action.point.x, curPoint.y + action.point.y));
        } else {
            pointAdjusted(action.point);
        }
        setBoost(action.boost);

    }

    public void quitDriver() {
        driver.quit();
    }


    private void delay(long period) {
        try {
            Thread.sleep(period);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void navigate() {
        driver.manage().window().setSize(WINDOW_SIZE);
        driver.manage().window().setPosition(new org.openqa.selenium.Point(0, 0));
        driver.get("http://slither.io");

        delay(2000);

        game = driver.findElement(By.cssSelector("body"));

        Point tl = getTopLeftPoint();
        gameRect = new Rectangle(tl.x, tl.y,
                WINDOW_SIZE.width - PIXELS_LEFT, WINDOW_SIZE.height - PIXELS_UP);

        boundsRect = new Rectangle(tl.x + 100, tl.y + 100,
                WINDOW_SIZE.width - PIXELS_LEFT - 100, WINDOW_SIZE.height - PIXELS_UP - 100);

        grabber = new FFmpegFrameGrabber("desktop");
        grabber.setOption("offset_x", String.valueOf(gameRect.x));
        grabber.setOption("offset_y", String.valueOf(gameRect.y));
        grabber.setFormat("gdigrab");
        grabber.setImageWidth(gameRect.width);
        grabber.setImageHeight(gameRect.height);
        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }


//        WebElement e = driver.findElement(By.id("nick"));
//        e.sendKeys("TEST-301");

//        delay(1000);

        driver.findElement(By.id("grqi")).click();

        driver.findElementByXPath("/html/body/div[@id='smh']" + // change skin button
                "/div[@id='cskh']/a[@id='csk']/img[@class='nsi']").click();
        delay(500);

        driver.findElementByXPath("/html/body/div[@class='btnt nsi sadg1']" + // save button
                "[2]/div/div[@class='nsi']").click();

        delay(500);
//        WebElement play = driver.findElements(By.className("nsi")).stream()
//                .filter(e1 -> e1.getText().equals(" Play ")).findFirst().orElseGet(null);
//        play.click();

        driver.findElementByXPath("/html/body/div[@id='login']" + // play button
                "/div[@id='playh']/div[@class='btnt nsi sadg1']/div/div[@class='nsi']").click();

        delay(5000);

        String[] elementsToHide = new String[]{"/html/body/div[9]", "/html/body/div[10]",
                "/html/body/div[11]", "/html/body/div[12]", "/html/body/div[13]",
                "/html/body/div[15]"};

        Arrays.stream(elementsToHide).map(driver::findElementByXPath).forEach(this::hideElement);
    }

    public void setBoost(boolean boost) {
        if (boost) {
            robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
        } else {
            robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
        }
    }

    private int attemptScoreRetrieval() throws StaleElementReferenceException {
        return Integer.valueOf(driver.findElementByXPath("/html[1]/body[1]/div[13]/span[1]/span[2]")
                .getAttribute("textContent"));
    }

    public int getScore() {
        int score = 0;
        boolean retrieved = false;
        int attempts = 0;

        while (!retrieved) {
            try {
                score = attemptScoreRetrieval();
                retrieved = true;
            } catch (StaleElementReferenceException e) {
                attempts++;

                if (attempts > 10) {
                    throw new StaleElementReferenceException("Failed to retrieve score after 10 attempts.");
                }
            }
        }

        return score;
    }

    private void hideElement(WebElement element) {
        driver.executeScript("arguments[0].setAttribute('style', arguments[0].getAttribute('style') + 'visibility:hidden;');",
                element);
    }


    public Frame getScreenshot() {
        try {
            return grabber.grabImage();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public Point getTopLeftPoint() {
        return new Point(game.getLocation().getX() + PIXELS_RIGHT, game.getLocation().getY() + PIXELS_DOWN);
    }

//    public Point getCenterPoint() {
//        return new Point((game.getLocation().getX() + PIXELS_RIGHT + game.getSize().width - PIXELS_LEFT) / 2,
//                (game.getLocation().getY() + PIXELS_DOWN + (game.getSize().height - PIXELS_UP) / 2));
//    }

    public java.awt.Point getMousePoint() {
        return new java.awt.Point(lastMouseX, lastMouseY);
    }

    public java.awt.Point roughly(java.awt.Point point) {
        return new java.awt.Point(Util.randInt(point.x - 2, point.x + 2), Util.randInt(point.y - 2, point.y + 2));
    }

    public void point(java.awt.Point point) {
        point = roughly(point);
        robot.mouseMove((int) point.getX(), (int) point.getY());
    }

    public void pointAdjusted(java.awt.Point point) {
        Point tl = getTopLeftPoint();
        point = roughly(point);
        robot.mouseMove((int) point.getX() + tl.x, (int) point.getY() + tl.y);
    }

    /**
     * @return true if the game ended and has not yet been restarted, false otherwise
     */
    public boolean testLoss() {
        List<WebElement> elem = driver.findElementsByXPath("/html[1]/body[1]/div[2]/div[5]/div[1]/div[1]/div[3]");

        if (elem.size() == 0) {
            return false;
        }

        return elem.get(0).isDisplayed();
    }

    /**
     * Restarts the game if it has ended. Does nothing otherwise.
     *
     * @return true if the game ended and is being restarted, false otherwise
     */
    public boolean fixLoss() {

        robot.mouseMove((int) gameRect.getCenterX(), (int) gameRect.getCenterY());
        robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);

        boolean lossDetected = false;
        try {
            WebElement play = driver.findElementByXPath("/html[1]/body[1]/div[2]/div[5]/div[1]/div[1]/div[3]");


            play.click();

            lossDetected = true;

            Thread.sleep(1000);

            WebElement ad = driver.findElementByXPath("/html[1]/body[1]/div[17]");
            ad.click();

        } catch (Exception e) {
            return lossDetected;
        }

        return true;
    }
}